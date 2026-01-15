// Copyright 2025 Jeroen van Nugteren

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// main header
#include "parfor.hh"

// local thread storage
#include <boost/thread/tss.hpp>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif


// namespace of RAT-Common
namespace rat { namespace cmn {

	// // declare the thread-local pointer directly in the .cpp file
	// boost::thread_specific_ptr<bool> thread_in_parfor;

	// // function to set the thread in parfor flag
	// void set_thread_in_parfor(bool value) {
	// 	if (!thread_in_parfor.get()) {
	// 		thread_in_parfor.reset(new bool(false));  // Initialize it if not already initialized
	// 	}
	// 	*thread_in_parfor = value;
	// }
	
	// // function to get the thread in parfor flag
	// bool get_thread_in_parfor() {
	// 	if (!thread_in_parfor.get()) {
	// 		return false;  // Default value if it's not set for this thread
	// 	}
	// 	return *thread_in_parfor;
	// }

	// get appropriate number of threads
	int get_num_threads() {
		int num_cpus = std::thread::hardware_concurrency();
		if(num_cpus>4)num_cpus--; // one cpu for the OS/GUI
		if (const char* str = std::getenv("RAT_NUM_THREADS")) {
			num_cpus = std::min(num_cpus, std::stoi(str));
		}
		return std::max(1, num_cpus);
	}

	// Helper class that wraps a thread pool and cleans it up.
	class ThreadPoolWrapper {
		// properties
		private:
			boost::asio::thread_pool pool_;
			boost::asio::executor_work_guard<boost::asio::thread_pool::executor_type> work_guard_;

		// methods
		public:
			// constructor
			ThreadPoolWrapper(unsigned num_threads)
				: pool_(num_threads),
				work_guard_(boost::asio::make_work_guard(pool_.get_executor()))

			{}

			// destructor
			~ThreadPoolWrapper() {
				// Reset the work guard, which allows the thread pool to exit its run loop.
				work_guard_.reset();

				// Stop the pool and wait for all threads to finish.
				pool_.stop();
				pool_.join();
			}

			// get
			boost::asio::thread_pool& get_thread_pool(){
				return pool_;
			}
	};

	// Global shared pointer for the thread pool wrapper.
	std::shared_ptr<ThreadPoolWrapper> global_threadpool_ptr;
	// Ensure thread-safe initialization.
	std::once_flag global_threadpool_once;

	// Access the global thread pool.
	boost::asio::thread_pool& get_thread_pool() {
		std::call_once(global_threadpool_once, []() {
			global_threadpool_ptr = std::make_shared<ThreadPoolWrapper>(get_num_threads());
		});
		return global_threadpool_ptr->get_thread_pool();
	}

	static int threads_available = get_num_threads();
	static std::mutex threads_mutex;

	int acquire_threads(const int num_requested) {
	    std::lock_guard<std::mutex> lock(threads_mutex);
	    // Provide up to the number of threads available.
	    int provided = std::min(num_requested, threads_available);
	    threads_available -= provided;
	    return provided;
	}

	void threads_finished(const int num_finished) {
	    std::lock_guard<std::mutex> lock(threads_mutex);
	    threads_available += num_finished;
	}

	int get_num_threads_available(){
		return threads_available;
	}

	// // access the threadpool
	// boost::asio::executor_work_guard<boost::asio::thread_pool::executor_type>& get_work_guard() {
	// 	static auto work_guard = boost::asio::make_work_guard(get_thread_pool().get_executor());
	// 	return work_guard;
	// }

	// // shut the whole thing down
	// void shutdown_thread_pool(){
	// 	// release the work guard so that the pool can finish when tasks complete.
	// 	get_work_guard().reset();
		
	// 	// optionally, call stop() if you want to cancel pending tasks:
	// 	get_thread_pool().stop();
		
	// 	// wait for all threads to finish their tasks.
	// 	get_thread_pool().join();
	// }

}}