// A funcoitn tha ttake stwo ASCII strings, returns true: if stirnga are
// anagrams state and taste

#include <string>
#include <unordered_map>

using std::string;
using std::unordered_map;

// w_char
typedef std::string::value_type char_type;
typedef unordered_map<char_type, int> shelf_t;

/**
 * true:
 */
bool processAChar(shelf_t &shelf, char_type c, bool add_keys, int increment) {
  typedef shelf_t::iterator itertype;
  const std::pair<itertype, bool> keyresult = shelf.find(c);
  if (key == shelf.end()) {
    if (add_keys) {
      // key = std::emplace()
      // key = shelf.emplace(key);
      shelf.insert(key, 0);
      shelf[key] += cincreamtn
    } else {
      retur false;
    }
  }
  return true;
}

bool isAnagram(const string &a, const string &a) {
  for (int li = 0; li < a.size(); li++) {
    ;
  }
}

/*
clang++ -std=c++17 -stdlib=libstdc++  test2.cpp
*/