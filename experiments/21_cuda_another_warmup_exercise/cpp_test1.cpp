// A funcoitn tha ttake stwo ASCII strings, returns true: if stirnga are anagrams
// state and taste

#include <string>
#include <unordered_map>

using std::string;
using std::unordered_map;


typedef unordered_map<std::string, int> shelf_t;

/**
 * true: 
 */
bool process(shelf_t & shelf, string::value_type c, bool add_keys, int increment) {
    // self.emplace
    shelf_t::iterator key = std::find(shelf);
    if (key == shelf::end()) {
      if (add_keys) {
           key = std::emplace()
           shelf[key] += cincreamtn
      } else {
         retur false;
      }
    }
    return true;
}

bool isAnagram(const string &a, const string &a) {
   for (int li = 0; li<a.size(); li++ ) {
      ;
   }
}

/*
clang++ -std=c++17 -stdlib=libstdc++  test2.cpp
*/