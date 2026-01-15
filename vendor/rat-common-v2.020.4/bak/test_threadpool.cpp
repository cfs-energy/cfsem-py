
#include <armadillo>
#include <list>

#include "threadpool.hh"
#include "log.hh"


int main(){
    // Create a thread pool with 4 worker threads
    rat::cmn::ShThreadPoolPr thread_pool = rat::cmn::ThreadPool::create(62);

    // create a log
    const rat::cmn::ShLogPr lg = rat::cmn::Log::create();

    // Enqueue tasks to calculate square roots
   	thread_pool->parfor(0,1024,true,[thread_pool](const arma::uword i, const arma::uword thread1){
   		thread_pool->parfor(0,1024,true,[thread1,i](const arma::uword j, const arma::uword thread2){
  			std::cout<<thread1<<" "<<thread2<<" "<<i<<" "<<j<<std::endl;
  		});
   	});


    return 0;
}