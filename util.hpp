#ifndef UTIL_H
#define UTIL_H

namespace util {
  ////////////////////////////////////////////////////////////
  // formated cout for the timings/parameters
  template <typename T>
  inline void pretty_print(const std::string& s, const T& d,
                           const std::string& unit = "",
			   std::ostream& os = std::cout){
    os.width(25); 
    os << s; 
    os.width(0);
    os << ": " << d << unit << std::endl;
  };
}

#endif
