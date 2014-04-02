#ifndef _COMMUNICATOR_H
#define _COMMUNICATOR_H

#include <Background.h>
#include <Geometry.hpp>
#include <Point.hpp>
#include <newQCDpt.h>
#include <IO.hpp>

#include <mpi.h>

#include <map>
#include <iostream>
#include <fstream>
#include <uparam.hpp>
#include <stdlib.h>



namespace kernels {
  template <class Field_t>
  struct base_types;
};


namespace comm {

  // Rules for communications
  namespace detail {
    
    ////////////////////////////////
    //
    // naming
    template<class M, int DIM>
    struct comm_types {
      typedef typename array_t<std::pair<int,int>, DIM>::Type nb_t;
      typedef typename std::vector< std::pair<std::vector<M>,
					      std::vector<M> > > buffer_t;
      typedef typename std::pair<MPI_Request,MPI_Request> request_t;
      typedef typename std::pair<MPI_Status,MPI_Status> status_t;
    }; 

    ////////////////////////////////
    //
    // MPI_Send wrapper
    template<class M, int DIM>
    struct Send : public comm_types<M,DIM> {
      typedef typename comm_types<M,DIM>::nb_t nb_t;
      typedef typename comm_types<M,DIM>::buffer_t buffer_t;
      typedef typename comm_types<M,DIM>::request_t request_t;
      const int send_sz = sizeof(M);

      Send(buffer_t& buffer, nb_t& neighbors) : buff(buffer), nb(neighbors) { }
      
      void operator()(const int& dir, const int& tag, std::vector<request_t>& r) {
	MPI_Isend(&buff[dir].first[0],buff[dir].first.size()*send_sz,
		  MPI_BYTE, nb[dir].first,tag, MPI_COMM_WORLD, &r[dir].first);
	MPI_Isend(&buff[dir].second[0],buff[dir].second.size()*send_sz,
		  MPI_BYTE, nb[dir].second,
		  tag+DIM, MPI_COMM_WORLD, &r[dir].second);	
      }
    private:
      buffer_t& buff;
      nb_t& nb;
    };

    ////////////////////////////////
    //
    // MPI_Recv wrapper
    template<class M, int DIM>
    struct Recv {
      typedef typename comm_types<M,DIM>::nb_t nb_t;
      typedef typename comm_types<M,DIM>::buffer_t buffer_t;
      typedef typename comm_types<M,DIM>::status_t status_t;
      const int send_sz = sizeof(M);
      
      Recv(buffer_t& buffer, nb_t& neighbors) : buff(buffer), nb(neighbors) { }

      void operator()(const int& dir, const int& tag, std::vector<status_t>& status) {
	MPI_Recv(&buff[dir].first[0],buff[dir].first.size()*send_sz,
		 MPI_BYTE, nb[dir].first, tag+DIM, MPI_COMM_WORLD,
		 &status[dir].first);
	MPI_Recv(&buff[dir].second[0],buff[dir].second.size()*send_sz,
		 MPI_BYTE, nb[dir].second, tag, MPI_COMM_WORLD,
		 &status[dir].second);
      }
    private:
      buffer_t& buff;
      nb_t& nb;
    };

  }



  template<class Field_t>
  struct Communicator : public kernels::base_types<Field_t> {

    typedef typename kernels::base_types<Field_t>::data_t data_t;
    typedef typename kernels::base_types<Field_t>::point_t Point;
    typedef typename kernels::base_types<Field_t>::direction_t Direction;
    static const int ORD = kernels::base_types<Field_t>::order;
    static const int DIM = kernels::base_types<Field_t>::n_dim;

    typedef typename array_t<std::pair<int,int>, DIM>::Type nb_t;
    typedef typename geometry::Geometry<DIM> Geometry;
    typedef typename Geometry::extents_t extents_t;
    typedef typename std::vector< std::pair<std::vector<data_t>,
					    std::vector<data_t> > > buffer_t;
    typedef typename std::vector<data_t>::iterator iterator_t;

    typedef typename std::pair<MPI_Request,MPI_Request> request_t;
    typedef typename std::pair<MPI_Status,MPI_Status> status_t;

    /// Buffer for communication
    buffer_t send_buff_;
    buffer_t recv_buff_;
    
    std::vector<request_t> request;
    std::vector<status_t> status;

    static nb_t nb_;
    static int numprocs_;
    static int rank_;
    static size_t storage_size;

    ////////////////////////////////
    //
    // Since we communicate after each timeslice the buffer size (for
    // directions != 0) is bnd_vol/T
    Communicator(const extents_t& e) {
      send_buff_.push_back
	(std::make_pair ( std::vector<data_t>
			  ( geometry::Geometry<DIM>(e).bnd_vol(0) ),
			  std::vector<data_t>
			  ( geometry::Geometry<DIM>(e).bnd_vol(0) ) ) );
      recv_buff_.push_back
	(std::make_pair ( std::vector<data_t>
			  ( geometry::Geometry<DIM>(e).bnd_vol(0) ),
			  std::vector<data_t>
			  ( geometry::Geometry<DIM>(e).bnd_vol(0) ) ) );
      request.push_back(request_t());
	status.push_back(status_t());
      for (int i = 1; i < DIM; ++i){
	send_buff_.push_back
	  (std::make_pair ( std::vector<data_t>
			    ( geometry::Geometry<DIM>(e).bnd_vol(i)/e[0] ),
			    std::vector<data_t>			        
			    ( geometry::Geometry<DIM>(e).bnd_vol(i)/e[0] ) ) );
	recv_buff_.push_back					        
	  (std::make_pair ( std::vector<data_t>			        
			    ( geometry::Geometry<DIM>(e).bnd_vol(i)/e[0] ),
			    std::vector<data_t>			        
			    ( geometry::Geometry<DIM>(e).bnd_vol(i)/e[0] ) ) );
	request.push_back(request_t());
	status.push_back(status_t());
      }

    }

    static void init(int argc, char *argv[]) {
      int rc = MPI_Init(&argc,&argv);
      if (rc != MPI_SUCCESS) {
	printf ("Error starting MPI program. Terminating.\n");
	MPI_Abort(MPI_COMM_WORLD, rc);
      }
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs_);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

      // if( rank_ == 0)
      // 	std::cout << "number of MPI processes: " << numprocs_ << std::endl;

      MPI_Barrier(MPI_COMM_WORLD);
      // int ff; 
      // MPI_Request rr;
      // MPI_Status ss;
      // MPI_Test(&rr, &ff, &ss); 
      // if(rank_ == 0)
      // 	std::cout << "request[3].status = " << (ff == MPI_SUCCESS) << std::endl;
      // MPI_Barrier(MPI_COMM_WORLD);

      // initialize neighbors *** only z direction ***
      nb_[0].first  = rank_;
      nb_[0].second = rank_;
      nb_[1].first  = rank_;
      nb_[1].second = rank_;
      nb_[2].first  = rank_;
      nb_[2].second = rank_;
      nb_[3].first  = (rank_+1)%numprocs_;
      nb_[3].second = (rank_+numprocs_-1)%numprocs_;     
      
      // if( rank_ == 1)
      // 	for(int mu(0); mu < DIM; ++mu)
      // 	    std::cout << "neighbour in direction " << mu
      // 		      << "+ is rank "              << nb_[mu].first
      // 		      << std::endl
      // 		      << "neighbour in direction " << mu
      // 		      << "- is rank "              << nb_[mu].second
      // 		      << std::endl
      // 		      << std::endl;

      MPI_Barrier(MPI_COMM_WORLD);

    };

    const int& rank() { return rank_; }
    const nb_t& nb() { return nb_; }
    const int& numprocs() { return numprocs_; }
    buffer_t& send_buff() { return send_buff_; }
    buffer_t& recv_buff() { return recv_buff_; }
    

    // Performs communication
    void do_it() {
      detail::Send<data_t,DIM> s(send_buff_, nb_);
      for( Direction mu(0); mu.is_good(); ++mu)
	if (nb_[int(mu)].second != rank_) {
	  s(int(mu),int(mu),request);
	}

      
      detail::Recv<data_t,DIM> r(recv_buff_, nb_);
      for( Direction mu(0); mu.is_good(); ++mu)
	if (nb_[int(mu)].first != rank_) {
	  r(int(mu),int(mu),status);
	  MPI_Barrier(MPI_COMM_WORLD);
	  
	}
      
      for( Direction mu(0); mu.is_good(); ++mu)
	if (nb_[int(mu)].first != rank_) {
	    MPI_Wait (&request[int(mu)].first,&status[int(mu)].first) ;
	    MPI_Wait (&request[int(mu)].second,&status[int(mu)].second) ;
	  }
      MPI_Barrier(MPI_COMM_WORLD);
      
    }

    static void finalize() {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
    }

    std::vector<request_t> test_request() { return request; }

    // Check if any communication is still on its way
    int test(const int& dir) {  
      int flag; 
      MPI_Test(&request[dir].first, &flag, &status[dir].first); 
      return flag;
    }

  };
  template<class Field_t> int Communicator<Field_t>::rank_;
  template<class Field_t> typename Communicator<Field_t>::nb_t Communicator<Field_t>::nb_;
  template<class Field_t> int Communicator<Field_t>::numprocs_;
  
  template<class T>
  struct Reduce {
    Reduce(const T& value) : x(value) {
      MPI_Comm_size(MPI_COMM_WORLD, &np_);      
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
      for(int i=0;i<np_;++i)
	rbuf.push_back(x);	
    };
    T& operator()() {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Gather( (void*)&x,sz_t,MPI_BYTE,&rbuf[0],sz_t,MPI_BYTE,0,MPI_COMM_WORLD);
      for(int i=1;i<np();++i)
      	rbuf[0] += rbuf[i];
      return rbuf[0];
    }
    const int& rank() { return rank_; }
    const int& np()   { return np_; }
  private:
    const int sz_t = sizeof(T);
    const T& x;
    int np_;
    int rank_;
    std::vector<T> rbuf;
  };

}


#endif //COMMUNICATOR_H

