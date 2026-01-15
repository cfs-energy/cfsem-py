// Copyright 2025 Jeroen van Nugteren

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following condition_s:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// include guard
#ifndef CMN_THREADPOOL_HH
#define CMN_THREADPOOL_HH

// headers
#include <armadillo>
#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <list>
#include <future>

// code specific to Rat
namespace rat{namespace cmn{

	// shared pointer definition
	typedef std::shared_ptr<class ThreadPool> ShThreadPoolPr;

	// threadpool class
	class ThreadPool {
		private:
			std::vector<std::thread> threads_;
			std::deque<std::function<void()> > task_queue_;
			std::mutex queue_mutex_;
			std::condition_variable condition_;
			bool stop_flag_;
			int num_tasks_running_;

		public:
			// constructor
			explicit ThreadPool(int num_threads = std::thread::hardware_concurrency()) : stop_flag_(false), num_tasks_running_(0){
				for (int i=0;i<num_threads;i++)
					threads_.emplace_back(std::bind(&ThreadPool::worker_thread, this));
			}

			// destructor
			~ThreadPool(){
				{
					std::unique_lock<std::mutex> lock(queue_mutex_);
					stop_flag_ = true;
				}

				condition_.notify_all();

				for (std::thread& thread : threads_) {
					thread.join();
				}
			}

			// create a pool
			static std::shared_ptr<ThreadPool> create(int num_threads = std::thread::hardware_concurrency()){
				return std::make_shared<ThreadPool>(num_threads);
			}

			// get total number of threads running on the pool
			int get_total_num_threads() const{
				return threads_.size();
			}

			// get number of tasks that is running
			int get_num_task_running()const{
				return num_tasks_running_;
			}

			// method for adding work to the task queue
			template <typename Function, typename... Args>
			auto enqueue(Function&& func, Args&&... args) -> std::future<decltype(func(args...))>{
				using ReturnType = decltype(func(args...));
				auto task = std::make_shared<std::packaged_task<ReturnType()>>(
					std::bind(std::forward<Function>(func), std::forward<Args>(args)...));			
				std::future<ReturnType> result = task->get_future();
				{
					std::unique_lock<std::mutex> lock(queue_mutex_);
					task_queue_.emplace_back([task]() { (*task)(); });
					num_tasks_running_++;
				}
				condition_.notify_one();
				return result;
			}

			// worker thread that executes tasks
			void worker_thread(){
				for(;;){
					std::function<void()> task;
					{
						std::unique_lock<std::mutex> lock(queue_mutex_);
						condition_.wait(lock, [this](){
							return stop_flag_ || !task_queue_.empty();
						});

						if (stop_flag_ && task_queue_.empty()) {
							return;
						}

						// get task from the deque
						if(!task_queue_.empty()){
							task = std::move(task_queue_.back());
							task_queue_.pop_back();
						}
					}

					// execute task
					if(task)task();

					{
						std::unique_lock<std::mutex> lock(queue_mutex_);
						num_tasks_running_--;
					}
				}
			}

			// // parfor loop running on the threadpool
			// template <class F>
			// void parfor(const arma::uword begin, const arma::uword end, bool enable_parallel, const F fn){
			// 	// Get the number of threads available in the pool
			// 	const int num_threads = threads_.size() - num_tasks_running_;

			// 	// use parallel for loop
			// 	if(enable_parallel && end>begin && num_threads>1) {
			// 		// create atomic index to keep track of iterate
			// 		std::atomic<arma::uword> idx(begin);

			// 		// create futures for the workers
			// 		std::vector<std::future<void>> futures(num_threads-1); // -1 to exclude the current thread

			// 		// create workers
			// 		for (int thread=0;thread<num_threads-1;thread++) {
			// 			futures[thread] = enqueue([&idx, &fn, thread, end]() {
			// 				for (;;) {
			// 					arma::uword i = idx.fetch_add(1);
			// 					if (i >= end) break;
			// 					fn(i, thread);
			// 				}
			// 			});
			// 		}

			// 		// join the workers to execute own task
			// 		for(;;){
			// 			arma::uword i = idx.fetch_add(1);
			// 			if (i >= end) break;
			// 			fn(i, num_threads - 1); // Use the last thread for the current thread
			// 		}

			// 		// Wait for stolen tasks to finish
			// 		for(auto& future : futures)future.wait();
			// 	}

			// 	// use normal for loop
			// 	else{
			// 		for(arma::uword i=begin;i<end;i++)fn(i, 0);
			// 	}
			// }

			// parallel-for template class
			template <class F>
			void parfor(arma::uword begin, arma::uword end, bool enable_parallel, F fn) {
				// make index
				std::atomic<arma::uword> idx;
				idx = begin;

				// fire asynchronous threads
				// don't run as threads when loop only consists of one walk
				if(enable_parallel==true && end-begin>0){
					// get number of cpu's
					int num_cpus = std::thread::hardware_concurrency();
					char* str = std::getenv("RAT_NUM_THREADS");
					if(str!=NULL)num_cpus = std::stoi(std::string(str));

					// calculate actual number of threads
					const int num_futures = int(std::min(arma::uword(num_cpus), end-begin+1));

					// get number of available CPU cores
					// int num_cpus = std::thread::hardware_concurrency();
					std::vector<std::future<void> > futures(num_futures);

					// walk over available cpus
					for (int cpu = 0; cpu != num_futures; cpu++) {
						futures[cpu] = std::async(std::launch::async,
							[cpu, &idx, end, &fn]() {
								// keep firing tasks untill run out of idx
								for (;;) {
									// atomic increment of index so that no two threads can do same task
									arma::uword i = idx++; 

									// check if last index
									if (i >= end) break;

									// run loop content with i
									fn(i, cpu);
								}
							}
						);
					}

					// make sure all threads are finished
					for (int cpu = 0; cpu != num_futures; cpu++) {
						futures[cpu].get();
					}
				}

				// just run as regular loop
				else{
					for(arma::uword i = begin;i<end;i++){
						fn(i,0);
					}
				}
			}
	};



}}

#endif