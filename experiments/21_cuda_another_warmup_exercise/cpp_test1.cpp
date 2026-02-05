// A funcoitn tha ttake stwo ASCII strings, returns true: if stirnga are
// anagrams state and taste

#include <iostream>
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
  itertype qq;

  std::cout << "processing char: " << c << "\n";

  // const std::pair<itertype, bool> keyresult = shelf.find(c);
  auto keyresult = shelf.find(c);
  // ^ why ref is returned??
  std::cout << "=========\n";

  std::cout << "??found?:?? " << keyresult->second << "\n";
  std::cout << "found?:?? " << keyresult->first << " , " << keyresult->second
            << "\n";

  // bool found = (keyresult != shelf.end());

  // why -> ?

  bool found = keyresult->second;
  if (found) {

    auto key1 = keyresult->first;
    std::cout << "found key: " << key1 << "\n";
    std::cout << "found?: " << keyresult->second << "\n";
    // /??
    // itertype key = key1;
  } else { // ~ (key == shelf.end())
    bool am_i_correct = (keyresult == shelf.end());
    std::cout << "am_i_correct: " << (am_i_correct ? "yes" : "no") << "\n";
    qq = shelf.end();

    if (add_keys) {
      char_type key = c;
      // key = std::emplace()
      // key = shelf.emplace(key);
      // shelf.insert(key, 0);
      shelf.emplace(key, 0);
      shelf[key] += increment;
    } else {
      return false;
    }
  }
  return true;
}

bool isAnagram(const string &a, const string &b) {
  shelf_t shelf;
  for (int li = 0; li < a.size(); li++) {
    bool result = processAChar(shelf, a[li], true, +1);
    std::cout << "processAChar a result: " << (result ? "yes" : "no") << "\n";
  }
  return false;
}

int main() {
  string a = "state";
  string b = "taste";
  bool result = isAnagram(a, b);
  std::cout << "are anagrams?: " << (result ? "yes" : "no") << "\n";
  return 0;
}

/*
clang++ -std=c++26 -stdlib=libstdc++  cpp_test1.cpp
*/