/*
* Key idea:
*
* In many contexts, an array decays into a pointer to its first element.
*
*/
int main(){
const char name[] = "J. P. Briggs";
// name's type is
// const char[13]
const char * ptrToName = name;
// array decays to pointer
}
void myFunc1(int param[]) {}
void myFunc2(int* param) {}
// same function as above
/*
* Key idea:
*
* Although functions can't declare parameters that are truly arrays, they can
* declare parameters that are references to arrays.
*
* The type deduced for T is the actual type of the array! That type includes
* the size of the array, so in this example T is deduced to be const
* char[13], and the type of f's parameter (a reference to this array) is
* const char (&)[13].
*
*/
template<typename T>
void f(T& param) {}
// template with by-reference parameter
int main(){
const char name[] = "J. P. Briggs";
// name's type is
// const char[13]
f(name);
// pass array to f
}
/*
* Key idea:
*
* Because array parameter declarations are treated as if they were pointer
* parameters, the type of an array that's passed to a template function by
* value is deduced to be a pointer type.
*/
template<typename T>
void f(T param) {}
// template with by-value parameter
int main(){
const char name[] = "J. P. Briggs";
// name's type is
// const char[13]
f(name);
// what types are deduced for T and param?
// -> name is array, but T deduced as const char*
}
/*
* Key idea:
*
* Considering the general form for templates and calls to it:
*
* template <typename T>
* void f(ParamType param);
*
* f(expr);
// deduce T and ParamType from expr
*
* then, in the simplest case when ParamType is a pointer type or a reference
* type, but not a universal reference, type deduction works like this:
*
* 1. If expr's type is a reference, ignore the reference part.
* 2. Then pattern-match expr's type against ParamType to determine T.
*
* If the type of f's parameter is changed from T& to const T&, the constness
* of cx and rx continues to be respected, but because we're now assuming that
* param is a reference-to-const, there's no longer a need for const to be
* deduced as part of T.
*/
template<typename T>
void f(const T& param) {}
// param is now a ref-to-const
int main(){
int x = 27;
// as before
const int cx = x;
// as before
const int& rx = x;
// as before
f(x);
// T is int, param's type is const int&
f(cx);
// T is int, param's type is const int&
f(rx);
// T is int, param's type is const int&
}
/*
* Key idea:
*
* Considering the general form for templates and calls to it:
*
* template <typename T>
* void f(ParamType param);
*
* f(expr);
// deduce T and ParamType from expr
*
* then, in the simplest case when ParamType is a reference type or a pointer
* type, but not a universal reference, type deduction works like this:
*
* 1. If expr's type is a reference, ignore the reference part.
* 2. Then pattern-match expr's type against ParamType to determine T.
*/
template<typename T>
void f(T& param) {}
// param is a reference
int main(){
int x = 27;
// x is an int
const int cx = x;
// cx is a const int
const int& rx = x;
// rx is a reference to x as a const int
f(x);
// T is int, param's type is int&
f(cx);
// T is const int,
// param's type is const int&
f(rx);
// T is const int,
// param's type is const int&
}
/*
* Key idea:
*
* Considering the general form for templates and calls to it:
*
* template <typename T>
* void f(ParamType param);
*
* f(expr);
// deduce T and ParamType from expr
*
* then, in the simplest case when ParamType is a pointer type or a reference
* type, but not a universal reference, type deduction works like this:
*
* * If expr's type is a reference, ignore the reference part.
* * Pattern-match expr's type against ParamType to determine T.
*
* If param were a pointer (or a pointer to const) instead of a reference,
* things would work essentially the same way:
*/
template<typename T>
void f(T* param) {}
// param is now a pointer
int main(){
int x = 27;
// as before
const int *px = &x;
// px is a ptr to x as a const int
f(&x);
// T is int, param's type is int*
f(px);
// T is const int,
// param's type is const int*
}
/*
* Key idea:
*
* Considering the general form for templates and calls to it:
*
* template <typename T>
* void f(ParamType param);
*
* f(expr);
// deduce T and ParamType from expr
*
* then, in the case when ParamType is a universal reference
* type, type deduction works like this:
*
* * If expr is an lvalue, both T and ParamType are deduced to be lvalue
* references
* * If expr is an rvalue, the usual type deduction rules apply.
*
*/
template<typename T>
void f(T&& param) {}
// param is now a universal reference
int main(){
int x = 27;
// as before
const int cx = x;
// as before
const int& rx = x;
// as before
f(x);
// x is lvalue, so T is int&,
// param's type is also int&
f(cx);
// cx is lvalue, so T is const int&,
// param's type is also const int&
f(rx);
// rx is lvalue, so T is const int&,
// param's type is also const int&
f(27);
// 27 is rvalue, so T is int,
// param's type is therefore int&&
}
/*
* Key idea:
*
* If we're dealing with pass-by-value
*
* template <typename T>
* void f(T param);
// param is now passed by value
*
* That means that param will be a copy of whatever is passed in - a
* completely new object. The fact that param will be a new object motivates
* the rules that govern how T is deduced from expr:
*
* 1. As before, if expr's type is a reference, ignore the reference part.
*
* 2. If, after ignoring expr's reference-ness, expr is const, ignore that,
* too. If it's volatile, also ignore that. (volatile objects are uncommon.
* They're generally used only for implementing device drivers.)
*/
template<typename T>
void f(T param) {}
// param is now passed by value
int main(){
int x = 27;
// as before
const int cx = x;
// as before
const int& rx = x;
// as before
f(x);
// T's and param's types are both int
f(cx);
// T's and param's types are again both int
f(rx);
// T's and param's types are still both int
const char* const ptr =
// ptr is const pointer to const object
"Fun with pointers";
f(ptr);
// pass arg of type const char * const
}
/*
* Key idea:
*
* The ability to declare references to arrays enables creation of a template
* to deduce the number of elements that an array contains.
*/
#include <array>
#include <cstddef>
// return size of an array as a compile-time constant. (The
// array parameter has no name, because we care only about
// the number of elements it contains.)
template<typename T, std::size_t N>
// see info
constexpr std::size_t arraySize(T (&)[N]) noexcept
// below on{
// constexpr
return N;
// and
}
// noexcept
int keyVals[] = { 1, 3, 7, 9, 11, 22, 35 };
// keyVals has
// 7 elements
int mappedVals1[arraySize(keyVals)];
// so does
// mappedVals
std::array<int, arraySize(keyVals)> mappedVals2;
// mappedVals'
// size is 7
/*
* Key-idea:
*
* Function types can decay into pointers, too, and everything regarding type
* deduction and arrays applies to type deduction for functions and their decay
* into function pointers.
*/
void someFunc(int, double){}
// someFunc is a function;
// type is void(int, double)
template<typename T>
void f1(T param) {}
// in f1, param passed by value
template<typename T>
void f2(T& param) {}
// in f2, param passed by ref
int main(){
f1(someFunc);
// param deduced as ptr-to-func;
// type is void (*)(int, double)
f2(someFunc);
// param deduced as ref-to-func;
// type is void (&)(int, double)
}
/*
* Key idea:
*
* If the function template looks like this:
*
* template <typename T>
* void f(ParamType param);
*
* then two types are deduced: one for T and one for ParamType. These types
* are frequently different, because ParamType often contains adornments,
* e.g., const- or reference qualifiers.
*/
template<typename T>
void f(const T& param) {}
// ParamType is const T&
int main(){
int x = 0;
f(x);
// call f with an int
}
/*
* Key idea:
*
* The treatment of braced initializers is the only way in which auto type
* deduction and template type deduction differ.
*/
#include <initializer_list>
template<typename T>
// template with parameter
void f(T param) {}
// declaration equivalent to
// x's declaration
template<typename T>
void f2(std::initializer_list<T> initList) {}
int main(){{
int x1 = 27;
int x2(27);
int x3 = {27};
int x4{27};
}{
auto x1 = 27;
// type is int, value is 27
auto x2(27);
// ditto
auto x3 = {27};
// type is std::initializer_list<int>,
// value is {27}
auto x4{27};
// ditto
//auto x5 = {1, 2, 3.0}; // error! can't deduce T for
// // std::initializer_list<T>
}{
auto x = { 11, 23, 9 };
// x's type is
// std::initializer_list<int>
//f({ 11, 23, 9 }); // error! can't deduce type for T
f2({ 11, 23, 9 });
// T deduced as int, and initList's
// type is std::initializer_list<int>
}
}
/*
* Key idea:
*
* Deducing types for auto is the same as deducing types for templates (with
* only one curious exception).
*/
template<typename T>
// conceptual template for
void func_for_x(T param) {}
// deducing x's type
template<typename T>
// conceptual template for
void func_for_cx(const T param) {}
// deducing cx's type
template<typename T>
// conceptual template for
void func_for_rx(const T& param) {}
// deducing rx's type
void someFunc(int, double) {}
// someFunc is a function;
// type is void(int, double)
int main(){
auto x = 27;
// case 3 (x is neither ptr nor reference)
const auto cx = x;
// case 3 (cx isn't either)
const auto& rx = x;
// case 1 (rx is a non-universal ref.)
auto&& uref1 = x;
// x is int and lvalue,
// so uref1's type is int&
auto&& uref2 = cx;
// cx is const int and lvalue
// so uref2's type is const int&
auto&& uref3 = 27;
// 27 is int and rvalue,
// so uref3's type is int&&
func_for_x(27);
// conceptual call: param's
// deduced type is x's type
func_for_cx(x);
// conceptual call: param's
// deduced type is cx's type
func_for_rx(x);
// conceptual call: param's
// deduced type is rx's type
const char name[] =
// name's type is const char[13]
"R. N. Briggs";
auto arr1 = name;
// arr1's type is const char*
auto& arr2 = name;
// arr2's type is
// const char (&)[13]
auto func1 = someFunc;
// func1's type is
// void (*)(int, double)
auto& func2 = someFunc;
// func2's type is
// void (&)(int, double)
}
/*
* Key ideas:
*
* 1. A function with an auto return type that returns a braced initializer list
* won't compile.
*
* 2. When auto is used in a parameter type specification in a C++14 lambda
* expression, things won't compile.
*/
#include <vector>
auto createInitList(){
//return {1, 2, 3}; // error: can't deduce type
// // for {1, 2, 3}
}
int main(){
std::vector<int> v;
auto resetV =
[&v](const auto& newValue) { v = newValue; };
// C++14
//resetV( {1, 2, 3} ); // error! can't deduce type
// // for { 1, 2, 3 }
}
/*
* Key idea:
*
* Sometimes, one needs decltype type deduction rules in cases where types are
* inferred. C++14 makes this possible through the decltype(auto) specifier:
* auto specifies that the type is to be deduced, and decltype says that
* decltype rules should be used during the deduction.
*/
void authenticateUser() {}
template<typename Container, typename Index>
// C++14 only;
decltype(auto)
// works, but
authAndAccess(Container& c, Index i)
// still requires{
// refinement
authenticateUser();
return c[i];
}
/*
* Key idea:
*
* The use of decltype(auto) is not limited to function return types. It can
* also be convenient for declaring variables when you want to apply the
* decltype type deduction rules to the initializing expression.
*/
class Widget {};
Widget w;
const Widget& cw = w;
auto myWidget1 = cw;
// auto type deduction:
// myWidget1's type is Widget
decltype(auto) myWidget2 = cw;
// decltype type deduction:
// myWidget2's type is
// const Widget&
/*
* Key idea:
*
* For lvalue expressions more complicated than names, decltype ensures that the
* type reported is always an lvalue reference. That is, if an lvalue
* expression other than a name has type T, decltype reports that type as T&.
*/
int main(){
int x = 0;
// decltype(x) is int
// decltype((x)) is int&
}
/*
* Key idea:
*
* This code attempts to assign 10 to an rvalue int, which is forbidden in C++,
* so the code won't compile.
*/
#include <deque>
//#include "compute_function_return_type_cpp14.h"
#include "compute_function_return_type_cpp11.h"
int main(){
std::deque<int> d;
authAndAccess(d, 5) = 10;
// authenticate user, return d[5],
// then assign 10 to it;
// this won't compile!
}
/*
* Key idea:
*
* In C++14, a seemingly trivial change in the way you write a return
* statement can affect the deduced type for a function.
*/
decltype(auto) f1(){
int x = 0;
// ...
return x;
// decltype(x) is int, so f1 returns int
}
decltype(auto) f2(){
int x = 0;
// ...
return (x);
// decltype((x)) is int&, so f2 returns int&
}
/*
* Key idea:
*
* Supporting such use means we need to revise the declaration for c to accept
* both lvalues and rvalues, and that means that c needs to be a universal
* reference.
*/
#include <string>
#include <deque>
#include "uref_cpp11.h"
std::deque<std::string> makeStringDeque()
// factory function{
std::deque<std::string> ds;
return ds;
}
int main(){
// make copy of 5th element of deque returned
// from makeStringDeque
auto s = authAndAccess(makeStringDeque(), 5);
}
/*
* Key idea:
*
* decltype almost always parrots back the type of the name or expression you
* give it without any modification.
*/
#include <cstddef>
#include <iostream>
class Widget {};
const int i = 0;
// decltype(i) is const int
bool f(const Widget& w);
// decltype(w) is const Widget&
// decltype(f) is bool(const Widget&)
struct Point {
int x, y;
// decltype(Point::x) is int
};
// decltype(Point::y) is int
Widget w;
// decltype(w) is Widget
template<typename T>
// simplified version of std::vector
class vector {
public:
// ...
T& operator[](std::size_t index);
// ...
};
int main(){
if (f(w)) {}
// decltype(f(w)) is bool
vector<int> v;
// decltype(v) is vector<int>
// ...
if (v[0] == 0) {}
// decltype(v[0]) is int&
}
/*
* Key idea:
*
* It is possible to produce accurate type information using Boost.TypeIndex.
*
*/
#include <iostream>
#include <typeinfo>
#include <vector>
#include <boost/type_index.hpp>
class Widget {};
template<typename T>
void f(const T& param){
using std::cout;
using boost::typeindex::type_id_with_cvr;
// show T
cout << "T = "
<< type_id_with_cvr<T>().pretty_name()
<< '\n';
// show param's type
cout << "param = "
<< type_id_with_cvr<decltype(param)>().pretty_name()
<< '\n';
}
std::vector<Widget> createVec()
// factory function{
std::vector<Widget> vw;
Widget w;
vw.push_back(w);
return vw;
}
int main(){
const auto vw = createVec();
// init vw w/factory return
if (!vw.empty()) {
f(&vw[0]);
// call f
// ...
}
}
/*
* Key idea:
*
* An effective way to get a compiler to show a type it has deduced is to use
* that type in a way that leads to compilation problems. The error message
* reporting the problem is virtually sure to mention the type that's causing
* it.
*/
template<typename T>
// declaration only for TD;
class TD;
// TD = "Type Displayer"
const int theAnswer = 42;
auto x = theAnswer;
auto y = &theAnswer;
//TD<decltype(x)> xType; // elicit errors containing
//TD<decltype(y)> yType; // x's and y's types
/*
* Key idea:
*
* Code editors in IDEs often show the types of program entities (e.g.
* variables, parameters, functions, etc.) when you do something like hover
* your cursor over the entity.
*/
const int theAnswer = 42;
auto x = theAnswer;
auto y = &theAnswer;
/*
* Key idea:
*
* Calls to std::type_info::name are not guaranteed to return anything
* sensible, but implementations try to be helpful. The level of helpfullnes
* varies.
*/
#include <iostream>
#include <typeinfo>
const int theAnswer = 42;
auto x = theAnswer;
auto y = &theAnswer;
int main(){
std::cout << typeid(x).name() << '\n';
// display types for
std::cout << typeid(y).name() << '\n';
// x and y
}
/*
* Key idea:
*
* The results of std::type_info::name are not reliable.
* Compiler diagnostics are a more dependable source of information about the
* results of type deduction.
*/
#include <iostream>
#include <typeinfo>
#include <vector>
class Widget {};
template<typename T>
// declaration only for TD;
class TD;
// TD = "Type Displayer"
template<typename T>
// template function to
void f(const T& param)
// be called{
using std::cout;
cout << "T = " << typeid(T).name() << '\n';
// show T
cout << "param = " << typeid(param).name() << '\n';
// show
// param's
// type
// TD<T> TType; // elicit errors containing
// TD<decltype(param)> paramType; // T's and param's types
}
std::vector<Widget> createVec()
// factory function{
std::vector<Widget> vw;
Widget w;
vw.push_back(w);
return vw;
}
int main(){
const auto vw = createVec();
// init vw w/factory return
if (!vw.empty()) {
f(&vw[0]);
// call f
// ...
}
}
/*
* Key idea:
*
* * Some advantages of using auto are:
* - avoidance of uninitialized variables
* - less verbose variable declarations
* - the ability to directly hold closures
* - the ability to avoid problems related to "type shortcuts"
*/
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include "Widget.h"
// auto variables have their type deduced from their initializer, so they must
// be initialized;
int x1;
// potentially uninitialized
//auto x2; // error! initializer required
auto x3 = 0;
// fine, x's value is well-defined
// Declaring a local variable whose value is that of a dereferenced iterator:
template<typename It>
// as before
void dwim(It b, It e){
while (b != e) {
auto currValue = *b;
// ...
}
}
// Because auto uses type deduction, it can represent types known only to
// compilers:
auto derefUPLess =
// comparison func.
[](const std::unique_ptr<Widget>& p1,
// for Widgets
const std::unique_ptr<Widget>& p2)
// pointed to by
{ return *p1 < *p2; };
// std::unique_ptr
// In C++14, parameters to lambda expressions may involve auto:
auto derefLess =
// C++14 comparison
[](const auto& p1,
// function for
const auto& p2)
// values pointed
{ return *p1 < *p2; };
// to by anything
// pointer-like
// Problems related to "type shortcuts":
std::vector<int> v;
//unsigned sz = v.size();
auto sz = v.size();
// sz's type is std::vector<int>::size_type
std::unordered_map<std::string, int> m;
int main(){
for (const std::pair<std::string, int>& p : m)
// Don't do this!{
// do something with p
}
for (const auto& p : m)
// Do this: more efficient and{
// easier to type!
// do something with p
}
}
/*
* Key idea:
*
* * Not using auto leads to certain inconveniences.
*/
#include <functional>
#include <memory>
#include "Widget.h"
// Simple joy 1: whether x gets initialized to zero or not, depends on the
// context.
int x;
// Simple joy 2: the simple joy of declaring a local variable to be initialized
// with the value of an iterator.
template<typename It>
// algorithm to dwim ("do what I mean")
void dwim(It b, It e)
// for all elements in range from{
// b to e
while (b != e) {
typename std::iterator_traits<It>::value_type
currValue = *b;
// ...
}
}
// Simple joy 3: the delight of declaring a local variable whose type is that
// of a closure. But the type of a closure is known only to the compiler, hence
// can't be written out...
// Perhaps you're thinking that we don't really need auto to declare a variable
// that holds a closure, because we use a std::function object:
std::function<bool(const std::unique_ptr<Widget>&,
const std::unique_ptr<Widget>&)>
derefUPLess = [](const std::unique_ptr<Widget>& p1,
const std::unique_ptr<Widget>& p2)
{ return *p1 < *p2; };
/*
* Key idea:
*
* The typed initializer idiom forces auto to deduce the type you want it to
* have.
*
* See also comments and discussions in [1-3].
*
* References:
*
* [1] http:
//www.aristeia.com/BookErrata/emc++-errata.html
* [2] http:
//stackoverflow.com/questions/25607216/why-should-i-prefer-the-explicitly-typed-initializer-idiom-over-explicitly-giv
* [3] https:
//herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/
*/
#include <vector>
double calcEpsilon()
// return tolerance value{
return 1.0; 
}
int main(){
/* Example 1 */{
float ep = calcEpsilon();
// implicitly convert
// double -> float
}{
// explicitly typed initializer idiom
auto ep = static_cast<float>(calcEpsilon());
}
/* Example 2 */
std::vector<int> c;
double d = 0.5;{
// this obscures the fact that d is converted to an int
int index = d*c.size();
}{
// explicitly typed initializer idiom
auto index = static_cast<int>(d*c.size());
}
}
/*
* Key idea:
*
* "Invisible" proxy types can cause auto to deduce the "wrong" type for an
* initializing expression.
*/
#include <iostream>
#include <vector>
class Widget {};
void processWidget(Widget& w, bool p){
std::cout << "processWidget(Widget&, bool)" << std::endl;
}
std::vector<bool> features(const Widget& w){
std::cout << "features(const Widget&)" << std::endl;
}
int main(){
Widget w;{
bool highPriority = features(w)[5];
// is w high priority?
processWidget(w, highPriority);
// process w in accord
// with its priority
}{
auto highPriority = features(w)[5];
// is w high priority?
processWidget(w, highPriority);
// undefined behavior!
// highPriority contains
// dangling pointer!
}{
// explicitly typed initializer idiom
auto highPriority = static_cast<bool>(features(w)[5]);
processWidget(w, highPriority);
}
}
/*
* TODO: work out this Matrix example, it currently doesn't compile.
*/
class Matrix;
template<typename T1, typename T2>
class Sum {
// TODO: implement implicit conversion from the proxy class to Matrix
operator Matrix() const{
return Matrix();
}
};
class Matrix {
public:
Sum<Matrix, Matrix> operator+(const Matrix& m1, const Matrix m2);
};
int main(){
Matrix m1, m2, m3, m4;
// typed initializer idiom
auto sum = static_cast<Matrix>(m1 + m2 + m3 + m4);
}
/*
* Key idea:
*
* * Empty braces mean no arguments, not an empty std::initializer_list.
*
* * If you want to call a std::initializer_list constructor with an
* empty std::initializer_list, you do it by making the empty braces
* a constructor argument - by putting the empty braces inside the
* parentheses or braces demarcating what you're passing.
*
* * For more discussion, see also blog post [1].
*
* References:
*
* [1] http:
//scottmeyers.blogspot.be/2016/11/help-me-sort-out-meaning-of-as.html
*/
#include <iostream>
#include <vector>
class Widget {
public:
Widget() {
// default ctor
std::cout << "Widget()" << std::endl;
};
Widget(std::initializer_list<int> il) {
// std::init_list ctor
std::cout << "Widget(std::initializer_list<int>)" << std::endl;
};
// ... // no implicit
// conversion funcs
};
int main(){
Widget w1;
// calls default ctor
Widget w2{};
// also calls default ctor
Widget w3();
// most vexing parse! declares a function!
Widget w4({});
// calls std::init_list ctor
// with empty list
Widget w5{{}};
// ditto
}
/*
* Extra code from the blog post [1]. This is the first version, which contained
* some bugs.
*
* References:
*
* [1] http:
//scottmeyers.blogspot.be/2016/11/help-me-sort-out-meaning-of-as.html
*/
#include <iostream>
#include <initializer_list>
class DefCtor {
public:
DefCtor(){}
//explicit DefCtor(){} // now explicit (gcc rejects this...)
};
class DeletedDefCtor {
public:
DeletedDefCtor() = delete;
};
class NoDefCtor {
public:
NoDefCtor(int){}
};
template<typename T>
class X {
public:
X() { std::cout << "Def Ctor\n"; }
X(std::initializer_list<T> il){
std::cout << "il.size() = " << il.size() << '\n';
}
};
int main(){
X<DefCtor> a0({});
// il.size = 0
X<DefCtor> b0{{}};
// il.size = 1
X<DeletedDefCtor> a2({});
// il.size = 0
X<DeletedDefCtor> b2{{}};
// il.size = 1
X<NoDefCtor> a1({});
// il.size = 0
X<NoDefCtor> b1{{}};
// il.size = 0
}
/*
* Extra code from the blog post [1]. This is the second version, with the
* bugs removed.
*
* References:
*
* [1] http:
//scottmeyers.blogspot.be/2016/11/help-me-sort-out-meaning-of-as.html
*/
#include <iostream>
#include <initializer_list>
class DefCtor {
int x;
public:
DefCtor(){}
//explicit DefCtor(){} // gcc rejects this...
};
class DeletedDefCtor {
int x;
public:
DeletedDefCtor() = delete;
};
class NoDefCtor {
int x; 
public:
NoDefCtor(int){}
};
template<typename T>
class X {
public:
X() { std::cout << "Def Ctor\n"; }
X(std::initializer_list<T> il){
std::cout << "il.size() = " << il.size() << '\n';
}
};
int main(){
X<DefCtor> a0({});
// il.size = 0
X<DefCtor> b0{{}};
// il.size = 1
X<DeletedDefCtor> a2({});
// il.size = 0
//X<DeletedDefCtor> b2{{}}; // error! attempt to use deleted constructor
X<NoDefCtor> a1({});
// il.size = 0
X<NoDefCtor> b1{{}};
// il.size = 0
}
/**
* Key ideas:
*
* * Initialization values may be specified in many different ways.
*
* * C++11 introduces 'uniform initialization' to address the confusion of
* multiple initialization syntaxes, as well as the fact that they don't
* cover all initialization scenarios.
*
* * A feature of braced initialization is that it prohibits implicit
* narrowing conversions among built-in types.
*/
#include <atomic>
#include <vector>
class Widget {
public:
Widget() {};
Widget(int x) {};
Widget(int i, bool b) {};
// ctors not declaring
Widget(int i, double d) {};
// std::initializer_list params
// Default initialization values for non-static data members.
private:
int x{0};
// fine, x's default value is 0
int y = 0;
// also fine
//int z(0); // error!
};
int main(){{
int x(0);
// initializer is in parentheses
int y = 0;
// initializer follows "="
int z {0};
// initializer is in braces
}{
int z = {0};
// initializer uses "=" and braces
}{
Widget w1;
// call default constructor
Widget w2 = w1;
// not an assignment; calls copy ctor
w1 = w2;
// an assignment; calls copy operator=
}
// Braced initialization{
std::vector<int> v{1, 3, 5};
// v's initial content is 1, 3, 5
}
// Uncopyable objects (e.g. std::atomics) may be initialized
// using braces or parentheses, but not using "=":{
std::atomic<int> ai1{0};
// fine
std::atomic<int> ai2(0);
// fine
//std::atomic<int> ai3 = 0; // error!
}
// Implicit narrowing conversions:
double x, y, z;
//int sum1{x + y + z}; // error! sum of doubles may
// not be expressible as int
// Initialization using parentheses and "=" doesn't check
// for narrowing conversions, because that would break too
// much legacy code:{
int sum2 = x + y + z;
// okay (value of expression
// truncated to an int)
int sum3(x + y + z);
// ditto
}{
Widget w1(10);
// call Widget ctor with argument 10
//Widget w2(); // most vexing parse! declares a function
// named w2 that returns a Widget!
void f(const Widget& w = Widget());
// w's default value is a
// default-constructed
// Widget
}{
Widget w3{};
// calls Widget ctor with no args
}{
void f(const Widget& w = Widget{});
// as before, w's default
// value is a default-
// constructed Widget
}{
auto v1 = -1;
// -1's type is int, and so is v1's
auto v2(-1);
// -1's type is int, and so is v2's
auto v3{-1};
// -1's type is still int, but
// v3's type is std::initializer_list<int>
auto v4 = {-1};
// -1's type remains int, but
// v4's type is std::initializer_list<int>
}
// In constructor calls, parentheses and braces have the same meaning as
// long as std::initializer_list parameters are not involved:{
Widget w1(10, true);
// calls first ctor
Widget w2{10, true};
// also calls first ctor
Widget w3(10, 5.0);
// calls second ctor
Widget w4{10, 5.0};
// also calls second ctor
}
}
/*
* Key idea:
*
* * In constructor calls, parentheses and braces have the same meaning as
* long as std::initializer_list parameters are not involved.
*/
#include <iostream>
class Widget {
public:
Widget(int i, bool b) { std::cout << "Widget(int, bool)" << std::endl; };
// ctors not declaring
Widget(int i, double d) { std::cout << "Widget(int, double)" << std::endl; };
// std::initializer_list params
};
int main(){
// In constructor calls, parentheses and braces have the same meaning as
// long as std::initializer_list parameters are not involved:
Widget w1(10, true);
// calls first ctor
Widget w2{10, true};
// also calls first ctor
Widget w3(10, 5.0);
// calls second ctor
Widget w4{10, 5.0};
// also calls second ctor
}
/*
* Key idea:
*
* * In constructor calls, parentheses and braces have the same meaning as
* long as std::initializer_list parameters are not involved.
*
* * If one or more constructors declare a parameter of type std::initializer_list,
* calls using the braced initialization syntax strongly prefer the overloads
* taking std::initializer_lists. Strongly. If there's any way for compilers
* to construe a call using braced initializer to be to a constructor taking a
* std::initializer_list, compilers will employ that interpretation.
*
* * Even what would normally be copy and move construction can be hijacked by
* std::initializer_list constructors.
*/
#include <iostream>
class Widget {
public:
Widget(int i, bool b) {
// as before
std::cout << "Widget(int, bool)" << std::endl;
};
Widget(int i, double d) {
// as before
std::cout << "Widget(int, double)" << std::endl;
};
Widget(std::initializer_list<long double> il) {
// added
std::cout << "Widget(std::initializer_list<long double>)" << std::endl;
};
operator float() const {
// convert to float
std::cout << "operator float() const" << std::endl;
}
};
int main(){{
Widget w1(10, true);
// uses parens and, as before,
// calls first ctor
Widget w2{10, true};
// uses braces, but now calls
// std::init_list ctor
// (10 and true convert to long double)
Widget w3(10, 5.0);
// uses parens and, as before,
// calls second ctor
Widget w4{10, 5.0};
// uses braces, but now calls
// std::initializer_list ctor
// (10 and 5.0 convert to long double)
Widget w5(w4);
// uses parens, calls copy ctor
Widget w6{w4};
// uses braces, calls
// std::initializer_list ctor
// (w4 converts to float, and float
// converts to long double)
Widget w7(std::move(w4));
// uses parens, calls move ctor
Widget w8{std::move(w4)};
// uses braces, calls
// std::initializer_list ctor
// (for same reason as w6)
}
}
/*
* Key idea:
*
* * Compiler's determiniation to match braced initializers with constructors
* taking std::initializer_lists is so strong, it prevails even if the
* best-match std::initializer_list constructor can't be called.
*/
#include <iostream>
class Widget {
public:
Widget(int i, bool b) {
// as before
std::cout << "Widget(int, bool)" << std::endl;
};
Widget(int i, double d) {
// as before
std::cout << "Widget(int double)" << std::endl;
};
Widget(std::initializer_list<bool> il) {
// element type is
// now bool
std::cout << "Widget(std::initializer_list<bool>)" << std::endl;
};
// ... // no implicit
// conversion funcs
};
int main(){
//Widget w{10, 5.0}; // error! requires narrowing conversions
}
/*
* Key idea:
*
* * Only if there's no way to convert the types of the arguments in a braced
* initializer to the type in a std::initializer_list do compilers fall back
* on normal overload resolution.
*/
#include <iostream>
class Widget {
public:
Widget(int i, bool b) {
// as before
std::cout << "Widget(int, bool)" << std::endl;
};
Widget(int i, double d) {
// as before
std::cout << "Widget(int double)" << std::endl;
};
// std::init_list element type is now std::string
Widget(std::initializer_list<std::string> il) {
// added
std::cout << "Widget(std::initializer_list<std::string>)" << std::endl;
};
// ... // no implicit 
// conversion funcs
};
int main(){{
Widget w1(10, true);
// uses parens, still calls first ctor
Widget w2{10, true};
// uses braces, now calls first ctor
Widget w3(10, 5.0);
// uses parens, still calls second ctor
Widget w4{10, 5.0};
// uses braces, now calls second ctor
}
}
/*
* Key idea:
*
* * If you're a template author, the tension between parentheses and braces
* for object creation can be especially frustrating, because, in general,
* it's not possible to know which should be used.
*/
#include <vector>
template<typename T,
// type of object to create
typename... Ts>
// types of arguments to use
void doSomeWork(Ts&&... params){
// create local T object from params...
// Method 1
T localObject(std::forward<Ts>(params)...);
// using parens
// Method 2
//T localObject{std::forward<Ts>(params)...}; // using braces
}
int main(){
std::vector<int> v;
doSomeWork<std::vector<int>>(10, 20);
}
/*
* Key idea:
*
* * If you create a std::vector of a numeric type (e.g., a std::vector<int>)
* and you pass two arguments to the constructor, whether you enclose those
* arguments in parentheses or braces makes a tremendous difference.
*/
#include <vector>
int main(){
std::vector<int> v1(10, 20);
// use non-std::initializer_list
// ctor: create 10-element
// std::vector, all elements have
// value of 20
std::vector<int> v2{10, 20};
// use std::initializer_list ctor:
// create 2-element std::vector,
// element values are 10 and 20
}
/*
* Key Idea:
*
* Using nullptr improves code clarity especially when auto variables are
* involved.
*/
int* findRecord() {
return nullptr;
}
int main(){{
auto result = findRecord(
/* arguments */ );
if (result == 0) {
}
}{
auto result = findRecord(
/* arguments */ );
if (result == nullptr) {
}
}
}
/*
* Key Idea:
*
* In C++98, passing 0 or NULL to pointer and integral overloads never calls
* the pointer overload. nullptr always calls the pointer overload.
*/
#include <iostream>
// Three overloads of f
void f(int) { std::cout << "f(int)" << std::endl; }
void f(bool) { std::cout << "f(bool)" << std::endl; }
void f(void*) { std::cout << "f(void*)" << std::endl; }
int main(){
f(0);
// calls f(int) overload, not f(void*)
//f(NULL); // might not compile, but typically calls
// f(int) overload. Never calls f(void*)
f(nullptr);
// calls f(void*) overload
}
/*
* Key Ideas:
*
* - nullptr shines especially brightly when templates enter the picture.
*
* - The following code for f1, f2, and f3 can be templatized.
*/
#include <iostream>
#include <memory>
#include <mutex>
class Widget {};
int f1(std::shared_ptr<Widget> spw) { std::cout << "f1" << std::endl; }
// call these only when
double f2(std::unique_ptr<Widget> upw) { std::cout << "f2" << std::endl; }
// the appropriate
bool f3(Widget* pw) { std::cout << "f3" << std::endl; }
// mutex is locked
int main(){
std::mutex f1m, f2m, f3m;
// mutexes for f1, f2, and f3
using MuxGuard =
// C++11 typedef; see Item 9
std::lock_guard<std::mutex>;{
MuxGuard g(f1m);
// lock mutex for f1
auto result = f1(0);
// pass 0 as null ptr to f1
}
// unlock mutex
// ...{
MuxGuard g(f2m);
// lock mutex for f2
auto result = f2(NULL);
// pass NULL as null ptr to f2
}
// unlock mutex
// ...{
MuxGuard g(f3m);
// lock mutex for f3
auto result = f3(nullptr);
// pass nullptr as null ptr to f3
}
// unlock mutex
}
/*
* Key Idea:
*
* lockAndCall is a templatized replacement for f1, f2, and f3.
* It demonstrates how template type deduction requires the null pointer
* argument to be nullptr, while 0 and NULL are deduced to integral type.
*/
#include <iostream>
#include <memory>
#include <mutex>
using MuxGuard =
// C++11 typedef; see Item 9
std::lock_guard<std::mutex>;
class Widget {};
int f1(std::shared_ptr<Widget> sp) { std::cout << "f1" << std::endl; }
// call these only when
double f2(std::unique_ptr<Widget> up) { std::cout << "f2" << std::endl; }
// the appropriate
bool f3(void* ptr) { std::cout << "f3" << std::endl; }
// mutex is locked
template<typename FuncType,
typename MuxType,
typename PtrType>
auto lockAndCall(FuncType func,
MuxType& mutex,
PtrType ptr) -> decltype(func(ptr)){
MuxGuard g(mutex);
return func(ptr);
}
int main(){
std::mutex f1m, f2m, f3m;
// mutexes for f1, f2, and f3
//auto result1 = lockAndCall(f1, f1m, 0); // error!
//auto result2 = lockAndCall(f2, f2m, NULL); // error!
auto result3 = lockAndCall(f3, f3m, nullptr);
// fine
}
/*
* Key Idea:
*
* In C++14, the trailing return type is replaced
* with decltype(auto) as its return type.
*/
#include <iostream>
#include <memory>
#include <mutex>
using MuxGuard =
// C++11 typedef; see Item 9
std::lock_guard<std::mutex>;
class Widget {};
int f1(std::shared_ptr<Widget> sp) { std::cout << "f1" << std::endl; }
// call these only when
double f2(std::unique_ptr<Widget> up) { std::cout << "f2" << std::endl; }
// the appropriate
bool f3(void* ptr) { std::cout << "f3" << std::endl; }
// mutex is locked
template<typename FuncType,
typename MuxType,
typename PtrType>
decltype(auto) lockAndCall(FuncType func,
// C++14
MuxType& mutex,
PtrType ptr){
MuxGuard g(mutex);
return func(ptr);
}
int main(){
std::mutex f1m, f2m, f3m;
// mutexes for f1, f2, and f3
//auto result1 = lockAndCall(f1, f1m, 0); // error!
//auto result2 = lockAndCall(f2, f2m, NULL); // error!
auto result3 = lockAndCall(f3, f3m, nullptr);
// fine
}
/*
* Key Idea:
*
* Using alias declarations is easier to read than function pointers.
*/
#include <string>
// FP is a synonym for a pointer to a function taking an int and
// a const std::string& and returning nothing
typedef void (*FP)(int, const std::string&);
// typedef
// same meaning as above
using FP = void (*)(int, const std::string&);
// alias
// declaration
/*
* Key Idea:
*
* Alias declarations may be templatized (typedefs cannot).
*/
#include <list>
#include <memory>
class Widget {};
template<typename T>
class MyAlloc : public std::allocator<T> {};
template<typename T>
// MyAllocList<T>
using MyAllocList = std::list<T, MyAlloc<T>>;
// is synonym for
// std::list<T,
// MyAlloc<T>>
MyAllocList<Widget> lw;
// client code
/*
* Key Idea:
*
* With alias declaration, dependent types no longer require
* to be preceded by typename (as does the cumbersome "::type" suffix).
*/
#include <list>
#include <memory>
template<typename T>
class MyAlloc : public std::allocator<T> {};
template<typename T>
using MyAllocList = std::list<T, MyAlloc<T>>;
// as before
template<typename T>
class Widget {
private:
MyAllocList<T> list;
// no "typename",
// ... // no "::type"
};
/*
* Key Idea:
*
* Typedefs cannot be templatized, and therefore need to be wrapped
* in a metafunction or container.
*/
#include <list>
#include <memory>
class Widget {};
template<typename T>
class MyAlloc : public std::allocator<T> {};
template<typename T>
// MyAllocList<T>::type
struct MyAllocList {
// is synonym for
typedef std::list<T, MyAlloc<T>> type;
// std::list<T,
};
// MyAlloc<T>>
MyAllocList<Widget>::type lw;
// client code
/*
* Key Idea:
*
* Typedefs cannot be templatized, and therefore need to be wrapped
* in a metafunction or container. In addition, dependent types
* must be preceded by typename.
*/
#include <list>
#include <memory>
template<typename T>
class MyAlloc : public std::allocator<T> {};
template<typename T>
// MyAllocList<T>::type
struct MyAllocList {
// is synonym for
typedef std::list<T, MyAlloc<T>> type;
// std::list<T,
};
// MyAlloc<T>>
template<typename T>
class Widget {
// Widget<T> contains
private:
// a MyAllocList<T>
typename MyAllocList<T>::type list;
// as a data member
// ...
};
/*
* Key Idea:
*
* MyAllocList<T>::type doesn't necessarily refer to a type. 
* Usage of a nested typedef renders it a dependent type, 
* (depending on what T is), and therefore the compiler requires
* preceding the template instantiation with typename.
*/
// TODO: get this to compile!
#include "linked_list_synonym_with_alias_template01.cpp"
class Wine { };
template<>
// MyAllocList specialization
class MyAllocList<Wine> {
// for when T is Wine
private:
enum class WineType
// see Item 10 for info on
{ White, Red, Blush };
// "enum class"
WineType type;
// in this class, type is
// ... // a data member!
};
/*
* Key Idea:
*
* If a new value is added to an enum that is included
* everywhere, the entire system will have to be recompiled.
*/
enum Status { good = 0,
failed = 1,
incomplete = 100,
corrupt = 200,
// audited = 500,
indeterminate = 0xFFFFFFFF
};
/*
* Key Idea:
*
* Forward declaration of enums removes the dependency
* on the enum definition.
*/
enum class Status;
// forward declaration
void continueProcessing(Status s);
// use of fwd-declared enum
/*
* Key Idea:
*
* Scoped enums can be forward-declared.
*/
//enum Color; // error!
enum class Color;
// fine
/*
* Key Idea:
*
* In C++11, the names of scoped enums do not belong to the scope containing
* the enum.
*/
enum class Color { black, white, red };
// black, white, red
// are scoped to Color
auto white = false;
// fine, no other
// "white" in scope
//Color c1 = white; // error! no enumerator named
// "white" is in this scope
Color c2 = Color::white;
// fine
auto c3 = Color::white;
// also fine (and in accord
// with Item4's advice)
/*
* Key Idea:
*
* Unscoped enums implicitly convert to integral types,
* allowing the below to run. Scoped enums are strongly
* typed.
*/
#include <cstddef>
#include <vector>
enum Color { black, white, red };
// unscoped enum
std::vector<std::size_t>
// func. returning
primeFactors(std::size_t x) {
// prime factors of x
std::vector<std::size_t> temp;
return temp;
}
int main(){
Color c = red;
// ...
if (c < 14.5) {
// compare Color to double (!)
auto factors =
// compute prime factors
primeFactors(c);
// of a color (!)
// ...
}
}
/*
* Key Idea:
*
* Scoped enums are strongly typed, and no implicit
* conversions are done. 
*/
#include <cstddef>
#include <vector>
enum class Color { black, white, red };
// enum is now scoped
std::vector<std::size_t>
// func. returning
primeFactors(std::size_t x) {
// prime factors of x
std::vector<std::size_t> temp;
return temp;
}
int main(){
Color c = Color::red;
// as before, but
// with scope qualifier
// if (c < 14.5) { // error! can't compare
// // Color and double
// auto factors = // error! can't pass Color to
// primeFactors(c); // function expecting std::size_t
// // ...
// }
}
/*
* Key Idea:
*
* For scoped enums, no implicit conversion is done. However
* type casting is still valid.
*/
#include <cstddef>
#include <vector>
enum class Color { black, white, red };
// enum is now scoped
std::vector<std::size_t>
// func. returning
primeFactors(std::size_t x) {
// prime factors of x
std::vector<std::size_t> temp;
return temp;
}
int main(){
Color c = Color::red;
// as before, but
// ... // with scope qualifier
if (static_cast<double>(c) < 14.5) {
// odd code, but
// it's valid
auto factors =
// suspect, but
primeFactors(static_cast<std::size_t>(c));
// it compiles
// ...
}
}
/*
* Key Idea:
*
* Every enum has an underlying type determined
* by the compiler - it may choose char, int, or
* any integral type.
*/
enum Color { black, white, red };
enum Status { good = 0,
failed = 1,
incomplete = 100,
corrupt = 200,
indeterminate = 0xFFFFFFFF
};
/*
* Key Idea:
*
* In C++11, all scoped enums have a default underlying type:
* int. It may also be specified explicity, can be forward
* declared, and the type specification can be placed in the
* definition.
*/
#include <cstdint>
enum class Status1;
// underlying type is int
enum class Status2: std::uint32_t;
// underlying type for
// Status is std::uint32_t
// (from <cstdint>)
enum Color: std::uint8_t;
// fwd decl for unscoped enum;
// underlying type is
// std::uint8_t
enum class Status3: std::uint32_t { good = 0,
failed = 1,
incomplete = 100,
corrupt = 200,
audited = 500,
indeterminate = 0xFFFFFFFF
};
/*
* Key Idea:
*
* In C++98 style enums, the names of these unscoped enumerators belong to the
* scope containing the enum, and that means nothing else in that scope may
* have the same name:
*/
enum Color { black, white, red };
// black, white, red are
// in same scope as Color
//auto white = false; // error! white already
// declared in this scope
/*
* Key Idea:
*
* Unscoped enums are useful in referencing std::tuple fields.
*/
#include <string>
#include <tuple>
using UserInfo =
// type alias; see Item 9
std::tuple<std::string,
// name
std::string,
// email
std::size_t> ;
// reputation
int main(){
UserInfo uInfo;
// object of tuple type
// ...
auto val = std::get<1>(uInfo);
// get value of field 1
}
/*
* Key Idea:
*
* Here an unscoped enum is used to reference a field in a
* std::tuple - an improvement to numbered fields.
*
* Since std::get requires a size_t, here we can take
* advantage of implicit conversion which would have otherwise
* be more effort using scoped enums.
*/
#include <string>
#include <tuple>
using UserInfo =
// type alias; see Item 9
std::tuple<std::string,
// name
std::string,
// email
std::size_t> ;
// reputation
int main(){
enum UserInfoFields { uiName, uiEmail, uiReputation };
UserInfo uInfo;
// as before
// ...
auto val = std::get<uiEmail>(uInfo);
// ah, get value of
// email field
}
/*
* Key Idea:
*
* The scoped enum method for referencing std::tuple
* field with an enum - requires a cast to size_t.
* Unscoped enums have an advantage here with implicit
* conversion.
*/
#include <string>
#include <tuple>
using UserInfo =
// type alias; see Item 9
std::tuple<std::string,
// name
std::string,
// email
std::size_t> ;
// reputation
int main(){
enum class UserInfoFields { uiName, uiEmail, uiReputation };
UserInfo uInfo;
// as before
// ...
auto val = std::get<static_cast<std::size_t>(UserInfoFields::uiEmail)>(uInfo);
}
/*
* Key Idea:
*
* Scoped enums require a cast to size_t type in order
* to reference a field in std::tuple. To create a helper
* function, use constexpr since std::get is a template.
*/
#include <string>
#include <tuple>
using UserInfo =
// type alias; see Item 9
std::tuple<std::string,
// name
std::string,
// email
std::size_t> ;
// reputation
template<typename E>
constexpr typename std::underlying_type<E>::type
toUType(E enumerator) noexcept{
return
static_cast<typename
std::underlying_type<E>::type>(enumerator);
}
int main(){
enum class UserInfoFields { uiName, uiEmail, uiReputation };
UserInfo uInfo;
// as before
// ...
auto val = std::get<toUType(UserInfoFields::uiEmail)>(uInfo);
}
/*
* Key Idea:
*
* Scoped enums require a cast to size_t type in order
* to reference a field in std::tuple. To create a helper
* function, use constexpr since std::get is a template.
*/
#include <string>
#include <tuple>
#include <type_traits>
using UserInfo =
// type alias; see Item 9
std::tuple<std::string,
// name
std::string,
// email
std::size_t> ;
// reputation
// Method 1
template<typename E>
// C++14
constexpr std::underlying_type_t<E>
toUType(E enumerator) noexcept{
return static_cast<std::underlying_type_t<E>>(enumerator);
}
// Method 2: using auto return type
//template<typename E> // C++14
//constexpr auto
// toUType(E enumerator) noexcept
//{
// return static_cast<std::underlying_type_t<E>>(enumerator);
//}
int main(){
enum class UserInfoFields { uiName, uiEmail, uiReputation };
UserInfo uInfo;
// as before
// ...
auto val = std::get<toUType(UserInfoFields::uiEmail)>(uInfo);
}
/*
* Key Idea:
*
* A technique to prevent calls with implicit conversions is
* to create deleted overloads for the types.
*/
bool isLucky(int number) { return true; }
// original function
bool isLucky(char) = delete;
// reject chars
bool isLucky(bool) = delete;
// reject bools
bool isLucky(double) = delete;
// reject doubles and
// floats
int main(){
//if (isLucky('a')) {} // is 'a' a lucky number?
// error! call to deleted function
//if (isLucky(true)) {} // is "true"?
// error!
//if (isLucky(3.5)) {} // should we truncate to 3
// before checking for luckiness?
// error!
//if (isLucky(3.5f)) {} // error!
}
/*
* Key Idea:
*
* Member function templates cannot be disabled by hiding them
* into private scope because it's a different access level.
*
* Deleted functions can be outside the class (in namespace scope).
*/
class Widget {
public:
// ...
template<typename T>
void processPointer(T* ptr)
{ }
private:
// template<> // error!
// void processPointer<void>(void*);
};
template<>
// still
void Widget::processPointer<void>(void*) = delete;
// public,
// but
// deleted
/*
* Key Idea:
*
* Deleted functions can be used to disable template instantiations.
* You can't do this using private member functions.
*/
template<typename T>
void processPointer(T* ptr) {};
template<>
void processPointer<void>(void*) = delete;
template<>
void processPointer<char>(char*) = delete;
template<>
void processPointer<const void>(const void*) = delete;
template<>
void processPointer<const char>(const char*) = delete;
class override {};
class Base {
public:
virtual ::override override(::override);
// override takes
// and returns an
};
// ::override
class Derived: public Base {
public:
::override override(::override) override;
// an override
};
// of above :-)
/*
* Key idea:
* The keyword override is reserved, but only in certain contexts. It has a
* reserved meaning only when it occurs at the end of a member function
* declaration. That means that if you have legacy code that already uses the
* name override, you don't need to change it for C++11.
*/
class Warning {
// potential legacy class from C++98
public:
void override();
// legal in both C++98 and C++11
// (with the same meaning)
};
/*
* Key idea:
*
* Virtual function overriding is what makes it possible to invoke a derived
* class function through a base class interface.
*/
#include <iostream>
#include <memory>
class Base {
public:
virtual void doWork() {
// base class virtual function
std::cout << "Base::doWork()" << std::endl;
}
};
class Derived : public Base {
public:
virtual void doWork() {
// overrides Base::doWork
std::cout << "Derived::doWork()" << std::endl;
// ("virtual" is optional
}
// here")
};
int main(){
std::unique_ptr<Base> upb =
// create base class pointer
std::make_unique<Derived>();
// to derived class object;
// see Item 21 for info on
// ... // std::make_unique
upb->doWork();
// call doWork through base
// class ptr; derived class
// function is invoked
}
/*
* Key idea:
*
* The need for reference-qualified member functions is not common, but it can
* arise.
*/
#include <vector>
class Widget {
public:
using DataType = std::vector<double>;
// see Item 9 for
// info on "using"
DataType& data() { return values; }
private:
DataType values;
};
Widget makeWidget(){
Widget w;
return w;
}
int main(){
Widget w;
auto vals1 = w.data();
// copy w.values into vals1
auto vals2 = makeWidget().data();
// copy values inside the
// Widget into vals2
}
/*
* Key idea
*
* The need for reference-qualified member functions is not common, but it can arise.
*/
#include <utility>
#include <vector>
class Widget {
public:
using DataType = std::vector<double>;
// see Item 9 for
// info on "using"
DataType& data() &
// for lvalue Widgets,
{ return values; }
// return lvalue
DataType&& data() &&
// for rvalue Widgets,
{ return std::move(values); }
// return rvalue
private:
DataType values;
};
Widget makeWidget(){
Widget w;
return w;
}
int main(){
Widget w;
auto vals1 = w.data();
// calls lvalue overload for
// Widget::data, copy-
// constructs vals1
auto vals2 = makeWidget().data();
// calls rvalue overload for
// Widget::data, move-
// constructs vals2
}
/*
* Key idea:
*
* Member function reference qualifiers are one of C++11's less-publicized
* features and make it possible to limit use of a member function to lvalues
* only or to rvalues only.
*/
#include <iostream>
class Widget {
public:
void doWork() & {
// this version of doWork applies only
std::cout << "doWork() &" << std::endl;
// when *this is an lvalue
}
void doWork() && {
// this version of doWork applies only
std::cout << "doWork() &&" << std::endl;
// when *this is an rvalue
}
};
Widget makeWidget()
// factory function (returns rvalue){
Widget w;
return w;
}
int main(){
Widget w;
// normal object (an lvalue)
w.doWork();
// calls Widget::doWork for lvalues
// (i.e., Widget::doWork &)
makeWidget().doWork();
// calls Widget::doWork for rvalues
// (i.e., Widget::doWork &&)
}
/*
* Key idea:
*
* The following code is completely legal and, at first sight, looks
* reasonable, but it contains no virtual function overrides - not a single
* derived class function that is tied to a base class function.
*/
class Base {
public:
virtual void mf1() const;
virtual void mf2(int x);
virtual void mf3() &;
void mf4() const;
};
class Derived: public Base {
public:
virtual void mf1();
virtual void mf2(unsigned int x);
virtual void mf3() &&;
void mf4() const;
};
/*
* Key idea:
*
* This the code-example that uses override and is correct.
*/
class Base {
public:
virtual void mf1() const;
virtual void mf2(int x);
virtual void mf3() &;
virtual void mf4() const;
};
class Derived: public Base {
public:
virtual void mf1() const override;
virtual void mf2(int x) override;
virtual void mf3() & override;
void mf4() const override;
// adding "virtual" is OK,
};
// but not necessary
/*
* Key idea:
*
* The below code won't compile, but, when written this way, compilers will
* kvetch about all the overriding-related problems.
*/
class Base {
public:
virtual void mf1() const;
virtual void mf2(int x);
virtual void mf3() &;
void mf4() const;
};
// Uncomment this, compile and see the compiler errors.
//class Derived: public Base {
//public:
// virtual void mf1() override;
// virtual void mf2(unsigned int x) override;
// virtual void mf3() && override;
// void mf4() const override;
//};
/*
* Key idea:
*
* In C++11, const_iterators are both easy to get and easy to use.
*/
#include <algorithm>
#include <vector>
int main(){
std::vector<int> values;
// as before
auto it =
// use cbegin
std::find(values.cbegin(), values.cend(), 1983);
// and cend
values.insert(it, 1998);
}
/*
* Key idea:
*
* About the only situation in which C++11's support for const_iterators comes
* up a bit short is when you want to write maximally generic library code.
*
* This works fine in C++14, but, sadly, not in C++11.
*/
#include <algorithm>
#include <iterator>
#include <vector>
template<typename C, typename V>
void findAndInsert(C& container,
// in container, find
const V& targetVal,
// first occurrence
const V& insertVal)
// of targetVal, then{
// insert insertVal
using std::cbegin;
// there
using std::cend;
auto it = std::find(cbegin(container),
// non-member cbegin
cend(container),
// non-member cend
targetVal);
container.insert(it, insertVal);
}
int main(){
std::vector<int> values;
// as before
findAndInsert(values, 1983, 1998);
}
/*
* Key idea:
*
* const_iterators were so much trouble in C++98.
*/
#include <algorithm>
#include <vector>
int main(){
std::vector<int> values;
// Approach 1: using iterators in C++98
std::vector<int>::iterator it =
std::find(values.begin(), values.end(), 1983);
values.insert(it, 1998);
// Approach 2: using const_iterators in C++98
typedef std::vector<int>::iterator IterT;
// type-
typedef std::vector<int>::const_iterator ConstIterT;
// defs
ConstIterT ci =
std::find(static_cast<ConstIterT>(values.begin()),
// cast
static_cast<ConstIterT>(values.end()),
// cast
1983);
//values.insert(static_cast<IterT>(ci), 1998); // may not
// compile; see
// below
}
/*
* Key idea
*
* TODO
*/
#include <vector>
class Widget {};
int main(){
std::vector<Widget> vw;
// ...
Widget w;
// ... // work with w
vw.push_back(w);
// add w to vw
// ...
}
/*
* Key idea
*
* Compilers typically offer no help in identifying inconsistencies between
* function implementations and their exception specifications.
*/
void setup() {};
void cleanup() {};
void doWork() noexcept{
setup();
// set up work to be done
// ... // do the actual work
cleanup();
// perform cleanup actions
}
int main(){
doWork();
}
/*
* Key idea:
*
* There is an additional incentive to apply noexcept to functions that won't
* produce exceptions: it permits compilers to generate better object code.
*/
#include <stdexcept>
#include <string>
int f(int x) throw();
// C++98 approach: f emits no
// exceptions
int f(int x) noexcept;
// C++11 approach: f emits no
// exceptions
int f(int x) noexcept
// C++98 version would use "throw()"{
if (x > 0) return x * x - 42;
// if x > 0 ...
throw std::invalid_argument(
// else throw!
"Invalid value for x: " + std::to_string(x)
);
}
/*
* Key ideas:
*
* - The constexpr in front of pow doesn't say that pow returns a const value,
* it says that if base and exp are compile-time constants, pow's result may
* be used as a compile-time constant. If base and/or exp are not
* compile-time constants, pow's result will be computed at runtime.
*
* - In C++11, constexpr functions may contain no more than a single
* executable statement: a return.
*/
#include <array>
int readFromDB(const char* s){
return 1;
}
constexpr
// pow's a constexpr func
int pow(int base, int exp) noexcept
// that never throws{
return (exp == 0 ? 1 : base * pow(base, exp - 1));
}
int main(){
// compile-time-compute the size of a std::array:
constexpr auto numConds = 5;
// # of conditions
std::array<int, pow(3, numConds)> results;
// results has
// 3^{numConds}
// elements
// runtime context:
auto base = readFromDB("base");
// get these values
auto exp = readFromDB("exponent");
// at runtime
auto baseToExp = pow(base, exp);
// call pow function
// at runtime
}
/*
* Key idea:
*
* - In C++14, the restrictions on constexpr functions are substantially
* looser.
*/
#include <array>
int readFromDB(const char* s){
return 1;
}
constexpr int pow(int base, int exp)
// C++14{
auto result = 1;
for (int i = 0; i < exp; ++i) result *= base;
return result;
}
int main(){
// compile-time-compute the size of a std::array:
constexpr auto numConds = 5;
// # of conditions
std::array<int, pow(3, numConds)> results;
// results has
// 3^{numConds}
// elements
// runtime context:
auto base = readFromDB("base");
// get these values
auto exp = readFromDB("exponent");
// at runtime
auto baseToExp = pow(base, exp);
// call pow function
// at runtime
}
/*
* Key idea:
*
* - The Point constructor can be declared constexpr, because if the arguments
* passed to it are known during compilation, the value of the data memberss
* of the constructed Point can also be known during compilation. Points so
* initialized could thus be constexpr.
*
* - The getters xValue and yValue can be constexpr, because if they're
* invoked on a Point object with a value known during compilation (e.g. a
* constexpr Point object), the values of the data members x and y can be
* known during compilation. That makes it possible to write constexpr
* functions that call Point's getters and to initialize constexpr objects
* with the results of such functions.
*/
#include "Point_cpp11.h"
constexpr Point
midpoint(const Point& p1, const Point& p2) noexcept{
return { (p1.xValue() + p2.xValue()) / 2,
// call constexpr
(p1.yValue() + p2.yValue()) / 2 };
// member funcs
}
int main(){
constexpr Point p1(9.4, 27.7);
// fine, "runs" constexpr
// ctor during compilation
constexpr Point p2(28.8, 5.3);
// also fine
constexpr auto mid = midpoint(p1, p2);
// init constexpr
// object w/result of
// constexpr function
}
/*
* Key idea:
*
* In C++14, Point's setters can be constexpr, making it possible
* to write the reflection function and have client code as given
* in the last statement of the main() function below.
*/
#include "Point_cpp14.h"
constexpr Point
midpoint(const Point& p1, const Point& p2) noexcept{
return { (p1.xValue() + p2.xValue()) / 2,
// call constexpr
(p1.yValue() + p2.yValue()) / 2 };
// member funcs
}
// return reflection of p with respect to the origin (C++14)
constexpr Point reflection(const Point& p) noexcept{
Point result;
// creat non-const Point
result.setX(-p.xValue());
// set its x and y values
result.setY(-p.yValue());
return result;
// return copy of it
}
int main(){
constexpr Point p1(9.4, 27.7);
// as above
constexpr Point p2(28.8, 5.3);
// also fine
constexpr auto mid = midpoint(p1, p2);
constexpr auto reflectedMid =
// reflectedMid's value is
reflection(mid);
// (-19.1 -16.5) and known
// during compilation
}
/*
* Key ideas:
*
* - integral values that are constant and known during compilation can be
* used in contexts where C++ requires an integral constant expression. Such
* contexts include specification of array sizes, integral template arguments
* (including lengths of std::array objects), enumerator values, alignment
* specifiers, and more. If you want to use a variable for these kind of
* things, you certainly want to declare it constexpr, because then compilers
* will ensure that it has compile-time value.
*
* - const doesn't offer the same guarantee as constexpr, because const
* objects need not be initialized with values known during compilation.
*/
#include <array>
int main(){
int sz;
// non constexpr variable
//constexpr auto arraysize1 = sz; // error! sz's value not
// known at compilation
//std::array<int, sz> data1; // error! same problem
constexpr auto arraySize2 = 10;
// fine, 10 is a
// compile-time constant
std::array<int, arraySize2> data2;
// fine, arraySize2
// is constexpr
const auto arraySize = sz;
// fine, arraySize is
// const copy of sz
//std::array<int, arraySize> data; // error! arraySize's value
// not known at compilation
}
/*
* Key idea:
*
* In some situations, a mutex is overkill. For example, if all you're doing
* is counting how many times a member function is called, a std::atomic
* counter will often be a less expensive way to go.
*/
#include <atomic>
#include <cmath>
class Point {
// 2D point
public:
double distanceFromOrigin() const noexcept
// see Item 14{
// for noexcept
++callCount;
// atomic increment
return std::sqrt((x*x)+(y*y));
}
private:
mutable std::atomic<unsigned> callCount{ 0 };
double x, y;
};
#include "Widget1.h"
//#include "Widget2.h"
//#include "Widget3.h"
int main(){
Widget w;
w.magicValue();
}
/*
* Key idea:
*
* Inside roots, one or both of these threads might try to modify the data
* members rootsAreValid and rootVals. That means that this code could have
* different threads reading and writing the same memory without
* synchronization, and that's the definition of a data race. This code has
* undefined behavior.
*/
#include <thread>
#include "Polynomial2.h"
Polynomial p;
void func1(){
/* Thread 1 */
auto rootsOfP = p.roots();
}
void func2(){
/* Thread 2 */
auto valsGivingZero = p.roots();
}
int main(){
std::thread t1(func1);
std::thread t2(func2);
t1.join();
t2.join();
}
/*
* Key idea:
*
* Unless a class inherits a destructor that's already virtual, the only way
* to make a destructor virtual is to explicitly declare it that way. Often,
* the default implementation would be correct, and "=default" is a good way
* to express that.
* However, a user-declared destructor suppresses generation of the move operations, so
* if movability is to be supported, "=default" often finds a second
* application.
* Declaring the move operations disables the copy operations, so if
* copyability is also desired, one more round of "=default" does the job.
*/
class Base {
public
virtual ~Base() = default;
// make dtor virtual
Base(Base&&) = default;
// support moving
Base& operator=(Base&&) = default;
Base(const Base&) = default;
// support copying
Base& operator=(const Base&) = default;
};
/*
* Key idea:
*
* Assuming that the class declares no copy operations, no move operations,
* and no destructor, compilers will automatically generate these functions if
* they are used.
*/
#include <map>
#include <string>
class StringTable {
public:
StringTable() {}
// ... // functions for insertion, erasure, lookup,
// etc., but no copy/move/dtor functionality
private:
std::map<int, std::string> values;
};
/*
* Key idea:
*
* Suppose it's decided that logging the default construction and destruction
* of such objects would be useful, and you add a destructor. Then declaring
* a destructor has a potentially significant side effect: it prevents the
* move operations from being generated. However, creation of the class's
* copy operations is unaffected. This can lead to a significant performance problem!
*/
#include <map>
#include <string>
void makeLogEntry(std::string s) {}
class StringTable {
public:
StringTable()
{ makeLogEntry("Creating StringTable object"); }
// added
~StringTable()
// also
{ makeLogEntry("Destroying StringTable object"); }
// added
// ... // other funcs as before
private:
std::map<int, std::string> values;
// as before
};
/*
* Key idea:
*
* As of C++11, the special member functions club has two more inductees: the
* move constructor and the move assignment operator. Their signatures are
* given here.
*/
class Widget {
public:
// ...
Widget(Widget&& rhs);
// move constructor
Widget& operator=(Widget&& rhs);
// move assignment operator
// ...
};
/*
* Key idea:
*
* C++11 deprecates the automatic generation of copy operations for classes
* declaring copy operations or a destructor. This means that if you have
* code that depends on the generation of copy operations in classes declaring
* a destructor or one of the copy operations, you should consider upgrading
* these classes to eliminate the dependence. C++11's "=default" lets you say
* that explicitly.
*/
class Widget {
public:
// ...
~Widget();
// user-declared dtor
// ... // default copy-ctor
Widget(const Widget&) = default;
// behavior is OK
Widget&
// default copy-assign
operator=(const Widget&) = default;
// behavior is OK
// ...
};
/*
* Key idea:
*
* There's nothing in the rules about the existence of a member function
* template preventing compilers from generating the special member functions.
*
* In the below case, compilers will still generate copy and move operations
* for Widget (assuming the usual conditions governing their generation are
* fulfilled), even though these templates could be instantiated to produce
* the signatures for the copy constructor and copy assignment operator. (That
* would be the case when T is Widget.)
*/
class Widget {
public:
// ...
template<typename T>
// construct Widget
Widget(const T& rhs);
// from anything
template<typename T>
// assign Widget
Widget& operator=(const T& rhs);
// from anything
// ...
};
// TODO: finish this! 
#include "investment.h"
#include "utils.h"
#include <memory>
auto delInvmt1 = [](Investment* pInvestment)
// custom{
// deleter
makeLogEntry(pInvestment);
// as
delete pInvestment;
// stateless
};
// lambda
template<typename... Ts>
// return type
std::unique_ptr<Investment, decltype(delInvmt1)>
// has size of
makeInvestment(Ts&&... args);
// Investment*
void delInvmt2(Investment* pInvestment)
// custom{
// deleter
makeLogEntry(pInvestment);
// as function
delete pInvestment;
}
template<typename... Ts>
// return type is
std::unique_ptr<Investment,
// sizeof Investment*
void (*)(Investment*)>
// plus at least size
makeInvestment(Ts&&... args);
// of function pointer!
#include "investment.h"
#include "utils.h"
#include <memory>
auto delInvmt = [](Investment* pInvestment)
// custom{
// deleter
makeLogEntry(pInvestment);
// (a lambda
delete pInvestment;
// expression)
};
// Variant 1
//template<typename... Ts> // return std::unique_ptr
//std::unique_ptr<Investment> // to an object created
// makeInvestment(Ts&&... args); // from the given args
// Variant 2
template<typename... Ts>
// revised
std::unique_ptr<Investment, decltype(delInvmt)>
// return type
makeInvestment(Ts&&... args){
std::unique_ptr<Investment, decltype(delInvmt)>
// ptr to be
pInv(nullptr, delInvmt);
// returned
if ( needStock ){
pInv.reset(new Stock(std::forward<Ts>(args)...));
}
else if ( needBond ){
pInv.reset(new Bond(std::forward<Ts>(args)...));
}
else if ( needRealEstate ){
pInv.reset(new RealEstate(std::forward<Ts>(args)...));
}
return pInv;
}
int main (){
// ...
auto pInvestment =
// pInvestment is of type
makeInvestment(
/* arguments */ ); // std::unique_ptr<Investment>
// ...
std::shared_ptr<Investment> sp =
// converts std::unique_ptr
makeInvestment(
/* arguments */ ); // to std::shared_ptr
}
// destroy *pInvestment
/*
* Key idea:
*
* In C++14, the existence of function return type deduction (see Item 3)
* means that makeInvestment could be implemented in a simpler and more
* encapsulated fashion.
*/
#include "investment.h"
#include "utils.h"
#include <memory>
template<typename... Ts>
auto makeInvestment(Ts&&... args)
// C++14{
auto delInvmt = [](Investment* pInvestment)
// this is now{
// inside
makeLogEntry(pInvestment);
// make-
delete pInvestment;
// Investment
};
std::unique_ptr<Investment, decltype(delInvmt)>
// as
pInv(nullptr, delInvmt);
// before
if ( needStock )
// as before{
pInv.reset(new Stock(std::forward<Ts>(args)...));
}
else if ( needBond )
// as before{
pInv.reset(new Bond(std::forward<Ts>(args)...));
}
else if ( needRealEstate )
// as before{
pInv.reset(new RealEstate(std::forward<Ts>(args)...));
}
return pInv;
// as before
}
int main (){
// ...
auto pInvestment =
// pInvestment is of type
makeInvestment(
/* arguments */ ); // std::unique_ptr<Investment>
// ...
std::shared_ptr<Investment> sp =
// converts std::unique_ptr
makeInvestment(
/* arguments */ ); // to std::shared_ptr
}
// destroy *pInvestment
#include <memory>
#include <vector>
class Widget {
public:
void process();
};
std::vector<std::shared_ptr<Widget>> processedWidgets;
void Widget::process(){
// ... // process the widget
processedWidgets.emplace_back(this);
// add it to list of
}
// processed Widgets;
// this is wrong!
#include <memory>
#include <vector>
class Widget : public std::enable_shared_from_this<Widget> {
public:
void process();
};
std::vector<std::shared_ptr<Widget>> processedWidgets;
void Widget::process(){
// as before, process the Widget
// ...
// add std::shared_ptr to current object to processedWidgets
processedWidgets.emplace_back(shared_from_this());
}
/*
* Key ideas:
*
* 1) For std::unique_ptr, the type of the deleter is part of the type of the
* smart pointer. For std::shared_ptr, it's not.
*
* 2) The std::shared_ptr design is more flexible.
*
* 3) Constructing more than one std::shared_ptr from a single
* raw pointer results in undefined behavior.
*/
#include <iostream>
#include <memory>
#include <vector>
class Widget {
public:
Widget() { std::cout << "Widget(" << this << ")" << std::endl; }
~Widget() { std::cout << "~Widget(" << this << ")" << std::endl; }
};
void makeLogentry(Widget *pw) { std::cout << "Log entry for " << pw << "." << std::endl; }
auto loggingDel = [](Widget *pw)
// custom deleter{
// (as in Item 19)
makeLogentry(pw);
delete pw;
};
int main(){
/* For std::unique_ptr, the type of the deleter is part of the type of the
* smart pointer. For std::shared_ptr, it's not.
*/
std::cout << "Part 1: deleter type being part or not" << std::endl;
std::unique_ptr<
// deleter type is
Widget, decltype(loggingDel)
// part of ptr type
> upw(new Widget, loggingDel);
std::shared_ptr<Widget>
// deleter type is not
spw(new Widget, loggingDel);
// part of ptr type
/* The std::shared_ptr design is more flexible: the type of the custom deleter
* is not part of the shared_ptr type, so the following things are for example
* possible:
* 
* -> can be placed in containers of objects of the shared_ptr type
* -> they can be assigned to one another
* -> they can be passed to a function taking a parameter of
* type std::shared_ptr<Widget>.
* -> ...
* 
* None of these things can be done with std::unique_ptrs that differ in the
* types of their custom deleters, because the type of the custom deleter would
* affect the type of the std::unique_ptr.
*/
std::cout << "Part 2: std::shared_ptr design being more flexible" << std::endl;
auto customDeleter1 = [](Widget *pw) { makeLogentry(pw); delete pw; };
// custom deleters,
auto customDeleter2 = [](Widget *pw) { makeLogentry(pw); delete pw; };
// each with a
// different type
std::shared_ptr<Widget> pw1(new Widget, customDeleter1);
std::shared_ptr<Widget> pw2(new Widget, customDeleter2);
std::vector<std::shared_ptr<Widget>> vpw { pw1, pw2 };
/*
* BAD BAD BAD: constructing more than one std::shared_ptr from a single
* raw pointer results in undefined behavior.
*/
std::cout << "Part 3: std::shared_ptr construction error" << std::endl;
//auto pw = new Widget;
// 
//std::shared_ptr<Widget> spw1(pw, loggingDel); // create control
// // block for *pw
//
//std::shared_ptr<Widget> spw2(pw, loggingDel); // create 2nd
// // control block
// // for *pw!
std::shared_ptr<Widget> spw1(new Widget,
// direct use of new
loggingDel);
std::shared_ptr<Widget> spw2(spw1);
// spw2 uses same
// control block as spw1
std::cout << "Part 4: right before end of main" << std::endl;
} 
/*
* Key idea:
*
* Use std::weak_ptr for std::shared_ptr-like pointers that can dangle.
*/
#include <memory>
#include <unordered_map>
class Widget {};
//class WidgetID {};
typedef int WidgetID;
std::unique_ptr<const Widget> loadWidget(WidgetID id){
return nullptr;
}
std::shared_ptr<const Widget> fastLoadWidget(WidgetID id){
static std::unordered_map<WidgetID,
std::weak_ptr<const Widget>> cache;
auto objPtr = cache[id].lock();
// objPtr is std::shared_ptr
// to cached object (or null
// if object's not in cache)
if (!objPtr) {
// if not in cache,
objPtr = loadWidget(id);
// load it
cache[id] = objPtr;
// cache it
}
return objPtr;
}
int main(){
auto spw =
// after spw is constructed,
std::make_shared<Widget>();
// the pointed-to Widget's
// ref count (RC) is 1. (See
// Item 21 for info on
// std::make_shared.)
// ...
std::weak_ptr<Widget> wpw(spw);
// wpw points to same Widget
// as spw. RC remains 1
// ...
spw = nullptr;
// RC goes to 0, and the
// Widget is destroyed.
// wpw now dangles
if (wpw.expired()) {
// if wpw doesn't point
}
// to an object...
std::shared_ptr<Widget> spw1 = wpw.lock();
// if wpw's expired,
// spw1 is null
auto spw2 = wpw.lock();
// same as above,
// but uses auto
std::shared_ptr<Widget> spw3(wpw);
// if wpw's expired,
// throw std::bad_weak_ptr
}
/*
* Key ideas:
*
* - Using std::make_unique and std::make_shared is preferable because you
* don't have to repeat the type begin created.
*
* - Using std::make_unique and std::make_shared is more exception safe.
*
* - A special feature of std::make_shared is improved efficiency.
*/
// TODO: this doesn't compile yet, make it to compile!
#include <iostream>
#include <memory>
#include <vector>
class Widget {};
class ReallyBigType {
/* ... */ };
void processWidget(std::shared_ptr<Widget> spw, int priority){
std::cout << "Processing Widget... done." << std::endl;
}
int computePriority(){
std::cout << "Computing priority... done." << std::endl;
return 1;
}
void cusDel(Widget *ptr)
// custom{
// deleter
std::cout << "Custom deleter." << std::endl;
}
int main(){
// Limitation 1: none of the make functions permit the specification of custom deleters.
auto widgetDeleter = [](Widget* pw) {
/* ... */ };
std::unique_ptr<Widget, decltype(widgetDeleter)>
upw(new Widget(), widgetDeleter);
std::shared_ptr<Widget> spw(new Widget(), widgetDeleter);
// Limitation 2: within the make function, the perfect forwarding code uses
// parentheses, not braces. The bad news of this is that if you want to
// construct your pointed-to object using a braced initializer, you must use
// new directly.{
auto upv = std::make_unique<std::vector<int>>(10,20);
auto spv = std::make_shared<std::vector<int>>(10,20);
}{
// create std::initializer_list
auto initList = {10, 20};
// create std::vector using std::initializer_list ctor
auto spv = std::make_shared<std::vector<int>>(initList);
}
// Limitation 3 (only for make_shared){
auto pBigObj =
// create very large
std::make_shared<ReallyBigType>();
// object via
// std::make_shared
// create std::shared_ptrs and std::weak_ptrs to
// large object, use them to work with it
// final std::shared_ptr to object destroyed here,
// but std::weak_ptrs to it remain
// during this period, memory formerly occupied
// by large object remains allocated
// final std::weak_ptr to object destroyed here;
// memory for control block and object is released
}{
std::shared_ptr<ReallyBigType> pBigObj(new ReallyBigType);
// create very large
// object via new
// as before, create std::shared_ptrs and
// std::weak_ptrs to object, use them with it
// final std::shared_ptr to object destroyed here,
// but std::weak_ptrs to it remain
// memory for object is deallocated
// during this period, only memory for
// the control block remains allocated
// final std::weak_ptr to object destroyed here;
// memory for control block and object is released
}
// Exception-unsafe call
processWidget(
// as before
std::shared_ptr<Widget>(new Widget, cusDel),
// potential
computePriority()
// resource
);
// leak!
// Exception-safe calls{
std::shared_ptr<Widget> spw(new Widget, cusDel);
processWidget(spw, computePriority());
// correct, but not
// optimal; see below
processWidget(std::move(spw),
// both efficient and
computePriority());
// exception safe
}
}
/*
* Key ideas:
*
* - Using std::make_unique and std::make_shared is preferable because you
* don't have to repeat the type begin created.
*
* - Using std::make_unique and std::make_shared is more exception safe.
*
* - A special feature of std::make_shared is improved efficiency.
*/
#include <iostream>
#include <memory>
#include <vector>
class Widget {};
void processWidget(std::shared_ptr<Widget> spw, int priority){
std::cout << "Processing Widget... done." << std::endl;
}
int computePriority(){
std::cout << "Computing priority... done." << std::endl;
}
int main(){
// Reason 1: no need to type the type twice.
auto upw1(std::make_unique<Widget>());
// with make func
std::unique_ptr<Widget> upw2(new Widget());
// without make func
auto spw1(std::make_shared<Widget>());
// with make func
std::shared_ptr<Widget> spw2(new Widget());
// without make func
// Reason 2: using the make_* versions is more exception safe.
processWidget(std::shared_ptr<Widget>(new Widget),
// potential
computePriority());
// resource
// leak!
processWidget(std::make_shared<Widget>(),
// no potential
computePriority());
// resource leak
// Reason 3: a special feature of make_shared is improved efficiency.{
std::shared_ptr<Widget> spw(new Widget);
}{
auto spw = std::make_shared<Widget>();
}
}
/*
* Key idea:
*
* std::make_unique is not part of C++11, but a basic version of it is easy
* to write yourself.
*/
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args){
return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
/*
* Key Idea:
*
* The Pimpl Idiom implemented C++98-style:
* all the data members are replaced with a pointer
* to struct defined here in the implementation file.
*/
#include "widget.h"
// in impl. file "widget.cpp"
#include "gadget.h"
#include <string>
#include <vector>
struct Widget::Impl {
// definition of Widget::Impl
std::string name;
// with data members formerly
std::vector<double> data;
// in Widget
Gadget g1, g2, g3;
};
Widget::Widget()
// allocate data members for
: pImpl(new Impl)
// this Widget object
{}
Widget::~Widget()
// destroy data members for
{ delete pImpl; }
// this object
#include "widget.h"
int main(){
Widget w1;
auto w2(std::move(w1));
// move-construct w2
w1 = std::move(w2);
// move-assign w1
}
/*
* Key Idea:
*
* std::shared_ptr doesn't require pointed-to types to be
* complete, and hence no special member functions need to
* be defined here.
*/
#include <memory>
#include <string>
// in "widget.cpp"
#include <vector>
#include "gadget.h"
#include "widget.h"
struct Widget::Impl {
// as before
std::string name;
std::vector<double> data;
Gadget g1, g2, g3;
};
Widget::Widget()
// per Item 22 create
: pImpl(std::make_shared<Impl>())
// std::unique_ptr
{}
// via std::make_unique
#include "widget.h"
int main(){
Widget w;
// error
}
/*
* Key Idea:
*
* Implementation of the Pimpl Idiom in C++11 -
* replaces the raw pointer with a smart pointer.
*
* Note that this compiles, but will fail for clients
* because the destructor is not implemented.
*/
#include "widget.h"
// in "widget.cpp"
#include "gadget.h"
#include <memory>
#include <string>
#include <vector>
struct Widget::Impl {
// as before, definition of
std::string name;
// Widget::Impl
std::vector<double> data;
Gadget g1, g2, g3;
};
Widget::Widget()
// per Item 22 create
: pImpl(std::make_unique<Impl>())
// std::unique_ptr
{}
// via std::make_unique
//Widget::~Widget() // ~Widget definition
//{}
Widget::~Widget() = default;
// same effect as above
Widget::Widget(Widget&& rhs) = default;
// defini-
Widget& Widget::operator=(Widget&& rhs) = default;
// tions
Widget::Widget(const Widget& rhs)
// copy ctor
: pImpl(std::make_unique<Impl>(*rhs.pImpl))
{}
Widget& Widget::operator=(const Widget& rhs)
// copy operator={
*pImpl = *rhs.pImpl;
return *this;
}
/*
* Key ideas:
*
* - text is not moved into value, it is copied.
* - two lessons learned:
* 1) don't declare objects const if you want to be able to move from
* them.
* 2) std::move not only doesn't actually move anything, it doesn't even
* guarantee that the object it's casting will be eligible to be moved.
*
*/
#include <string>
class Annotation {
public:
//explicit Annotation(std::string text); // param to be copied,
// // so per Item 41,
// // pass by value
explicit Annotation(const std::string text)
: value(std::move(text))
// "move" text into value; this code{
/* ... */ } // doesn't do what it seems to!
private:
std::string value;
};
/*
* Key idea:
*
* std::forward requires both a function argument (rhs.s) and a template type
* argument (std::string).
*/
#include <string>
class Widget {
public:
Widget(Widget&& rhs)
// unconventional,
: s(std::forward<std::string>(rhs.s))
// undesirable
{ ++moveCtorCalls; }
// implementation
private:
static std::size_t moveCtorCalls;
std::string s;
};
/*
* Key idea:
*
* - std::move's attractions are convenience, reduced likelihood of error, and
* greater clarity.
*
* - std::move requires only a functino argument (rhs.s)
*/
#include <string>
class Widget {
public:
Widget(Widget&& rhs)
: s(std::move(rhs.s))
{ ++moveCtorCalls; }
private:
static std::size_t moveCtorCalls;
std::string s;
};
/*
* Key idea:
*
* In essence, std::move casts its argument to an rvalue, and that's all it
* does.
*/
#include <type_traits>
using std::remove_reference;
template<typename T>
// in namespace std
typename remove_reference<T>::type&&
move(T&& param){
using ReturnType =
// alias declaration;
typename remove_reference<T>::type&&;
// see Item 9
return static_cast<ReturnType>(param);
}
/*
* Key idea:
*
* std::move can be easily implemented in C++14 thanks to function return type
* deduction and to the Standard Library's alias template
* std::remove_reference_t.
*/
#include <type_traits>
using std::remove_reference_t;
template<typename T>
// C++14 only; still
auto move(T&& param)
// in namespace std{
using ReturnType = remove_reference_t<T>&&;
return static_cast<ReturnType>(param);
}
#include "typical_use_of_std_forward.h"
#include <iostream>
void process(Widget& lvalArg) {
std::cout << "process(Widget&)" << std::endl;
}
void process(Widget&& rvalArg) {
std::cout << "process(Widget&&)" << std::endl;
}
int main(){
Widget w;
logAndProcess(w);
// call with lvalue
logAndProcess(std::move(w));
// call with rvalue
}
/*
* Key idea:
*
* If you see "T&&" without type deduction, you're looking at an rvalue
* reference.
*/
class Widget {};
void f(Widget&& param);
// no type deduction;
// param is an rvalue reference
Widget&& var1 = Widget();
// no type deduction;
// var1 is an rvalue reference
/*
* Key idea:
*
* auto universal references crop up quite a lot in C++14, because C++14
* lambda expressions may declare auto&& parameters.
*/
#include <utility>
auto timeFuncInvocation =
[](auto&& func, auto&&... params)
// C++14{
// start timer;
std::forward<decltype(func)>(func)(
// invoke func
std::forward<decltype(params)>(params)...
// on params
);
// stop timer and record elapsed time;
};
/*
* Key idea:
*
* It seems reasonable to assume that if you see "T&&" in source code, you're
* looking at an rvalue reference. However, it's not quite that simple.
*/
#include <vector>
class Widget {};
void f(Widget&& param);
// rvalue reference
Widget&& var1 = Widget();
// rvalue reference
auto&& var2 = var1;
// not rvalue reference
template<typename T>
void f(std::vector<T>&& param);
// rvalue reference
template<typename T>
void f(T&& param);
// not rvalue reference
/*
* Key idea:
*
* Universal references arise in two contexts.
*/
class Widget {};
Widget&& var1 = Widget();
// rvalue reference
// Context 1: function template parameters.
template<typename T>
void f(T&& param);
// param is a universal reference
// Context 2: auto declarations.
auto&& var2 = var1;
// var2 is a universal reference
/*
* Key idea:
*
* For a reference to be universal, type deduction is necessary, but it's not
* sufficient. The form of the reference declaration must also be correct,
* and that form is quite constrained.
*/
#include <utility>
#include <vector>
class Widget {};
template<typename T>
void f(std::vector<T>&& param);
// param is an rvalue reference
template<typename T>
void f(const T&& param);
// param is an rvalue reference
template<typename MyTemplateType>
// param is a
void someFunc(MyTemplateType&& param);
// universal reference
int main(){
std::vector<int> v;
// f(v); // error! can't bind lvalue to
// rvalue reference
}
// TODO: use Boost::TypeIndex instead
//#include "printtype.h"
#include <utility>
#include <iostream>
class Widget {};
template<typename T>
void f(T&& param)
// param is a universal reference{
// TODO: use Boost::TypeIndex instead
//std::cout << Printtype<decltype(param)>().name() << std::endl;
}
int main(){
Widget w;
f(w);
// lvalue passed to f; param's type is
// Widget& (i.e., an lvalue reference)
f(std::move(w));
// rvalue passed to f; param's type is
// Widget&& (i.e., an rvalue reference)
}
/*
* Key idea:
*
* Using std::move with universal references is a bad idea, because that can
* have the effect of unexpectedly modifying lvalues (e.g., local variables).
*/
#include <iostream>
#include <string>
std::string getWidgetName()
// factory function{
return std::string("SomeWidgetName");
}
class Widget {
public:
template<typename T>
void setName(T&& newName)
// universal reference
{ name = std::move(newName); }
// compiles, but is
// bad, bad, bad!
private:
std::string name;
//std::shared_ptr<SomeDataStructure> p; // REMARK: this is not really necessary for this example.
};
int main(){
Widget w;
auto n = getWidgetName();
// n is local variable
w.setName(n);
// moves n into w!
std::cout << n << std::endl;
// n's value now unknown
}
/*
* Key idea:
*
* Overloading for const lvalues and for rvalues could solve the problem,
* but there are drawbacks:
*
* - it's more sourcecode to write and maintain (two functions instead of
* a simple template).
*
* - it can be less efficient.
*
* - such a design has poor scalability: n parameters necessitates 2^n
* overloads.
*/
#include <iostream>
#include <string>
std::string getWidgetName()
// factory function{
return std::string("SomeWidgetName");
}
class Widget {
public:
void setName(const std::string& newName)
// set from
{ name = newName; }
// const lvalue
void setName(std::string&& newName)
// set from
{ name = std::move(newName); }
// rvalue
private:
std::string name;
//std::shared_ptr<SomeDataStructure> p; // REMARK: this is not really necessary for this example.
};
int main(){
Widget w;
auto n = getWidgetName();
// n is local variable
w.setName(n);
// moves n into w!
std::cout << n << std::endl;
// n's value now unknown
w.setName("Adela Novak");
}
/*
* Key idea:
*
* If you're in a function that returns by value, and you're returning an
* object bound to an rvalue reference or a universal reference, you'll want
* to apply std::move or std::forward when you return the reference.
*/
#include <utility>
class Matrix {
public:
Matrix& operator+=(const Matrix& rhs);
};
Matrix& Matrix::operator+=(const Matrix& rhs){
return *this;
}
Matrix
// by-value return
operator+(Matrix&& lhs, const Matrix& rhs){
lhs += rhs;
return std::move(lhs);
// move lhs into return value
//return lhs; // copy lhs into return value
}
/*
* Key idea:
*
* Here, we want to make sure that text's value doesn't get changed by
* sign.setText, because we want to use that value when we call
* signHistory.add. Ergo the use of std::forward on only the final use of the
* universal reference.
*/
// TODO:
// fix sign and signHistory
#include <chrono>
#include <utility>
template<typename T>
// text is
void setSignText(T&& text)
// univ. reference{
sign.setText(text);
// use text, but
// don't modify it
auto now =
// get current time
std::chrono::system_clock::now();
signHistory.add(now,
std::forward<T>(text));
// conditionally cast
}
// text to rvalue
/*
* Key idea:
*
* If the original object is an rvalue, its value should be moved into the
* return value (thus avoiding the expense of making a copy), but if the
* original value is an lvalue, an actual copy must be created.
*
* If the call to std::forward were omitted, frac would be unconditionally
* copied into reduceAndCopy's return value.
*/
#include <utility>
struct Fraction {
void reduce() {};
};
template<typename T>
Fraction
// by value return
reduceAndCopy(T&& frac)
// universal reference param{
frac.reduce();
return std::forward<T>(frac);
// move rvalue into return
}
// value, copy lvalue
/*
* Key idea:
*
* Given a function returning a local variable by value, it is not a good idea
* to turn the copy into a move in the return statement.
*
* The "copying" version of makeWidget can avoid the need to copy the local
* variable w by constructing it in the memory alloted for the function's
* return value. This is known as Return Value Optimization (RVO).
*
* The moving version of makeWidget does just what its name says it does
* (assuming Widget offers a move constructor): it moves the contents of w
* into makeWidget's return value location.
*
* Developers trying to help their compilers optimize by applying std::move to
* a local variable that's being returned are actually limiting the
* optimization options available to their compilers!
*
* When the RVO is permitted, either copy elision takes place or std::move is
* implicitly applied to local objects being returned.
*/
class Widget {};
Widget makeWidget(){
Widget w;
// local variable
// ... // configure w
return w;
// "copy" w into return value
//return std::move(w); // move w into return value
}
// (don't do this!)
Widget makeWidget(Widget w)
// by value parameter of same{
// type as function's return
// ...
return w;
// treat w as rvalue
}
/*
* Key ideas:
*
* rvalue references bind only to objects that are candidates for moving. If
* you have an rvalue reference parameter, you know that the object it's bound
* to may be moved.
*
* rvalue references should be unconditionally cast to rvalues (via std::move)
* when forwarding them to other functions, because they're always bound to
* rvalues.
*/
#include <memory>
#include <string>
class SomeDataStructure {};
class Widget {
public:
Widget(Widget&& rhs)
// rhs is rvalue reference
: name(std::move(rhs.name)),
p(std::move(rhs.p)){
/* ... */ }
private:
std::string name;
std::shared_ptr<SomeDataStructure> p;
};
/*
* Key ideas:
*
* A universal reference, might be bound to an object that's eligible for
* moving. Universal references should be cast to rvalues only if they were
* initialized with rvalues.
*
* Universal references should be conditionally cast to rvalues (via
* std::forward) when forwarding them, because they're only sometimes bound to
* rvalues.
*/
#include <memory>
#include <string>
class Widget {
public:
template<typename T>
void setName(T&& newName)
// newName is
{ name = std::forward<T>(newName); }
// universal reference
private:
std::string name;
};
/*
* Key idea:
*
* The perfect-forwarding constructor will generally match better than the
* copy constructor, which may produce compilation errors if the passed
* parameter is not of the same type as the member variable initialized. To
* force on using the copy constructor, you might pass a const lvalue.
*/
#include <iostream>
#include <string>
std::string nameFromIdx(int idx)
// return std::string{
// corresponding to idx
std::string s("Test");
return s;
}
class Person {
public:
//Person() = default;
template<typename T>
explicit Person(T&& n)
// perfect forwarding ctor;
: name(std::forward<T>(n)) {
// initializes data member
std::cout << "Person(T&&)" << std::endl;
}
explicit Person(int idx)
// int ctor
: name(nameFromIdx(idx)) {
std::cout << "Person(int idx)" << std::endl;
}
//Person(const Person& rhs); // copy ctor
// (compiler-generated)
//Person(Person&&); // move ctor
// (compiler-generated)
private:
std::string name;
};
//class SpecialPerson: public Person {
//public:
// SpecialPerson(const SpecialPerson& rhs) // copy ctor; calls
// : Person(rhs) // base class
// { /* ... */ } // forwarding ctor!
//
// SpecialPerson(SpecialPerson&& rhs) // move ctor; calls
// : Person(std::move(rhs)) // base class
// { /* ... */ } // forwarding ctor!
//};
int main(){{
Person p("Nancy");
//auto cloneOfP(p); // create new Person from p;
// this won't compile!
}{
const Person cp("Nancy");
// object is now const
auto cloneOfP(cp);
// calls copy constructor!
}
}
/*
* Key idea:
*
* This code is not unreasonable, but not as efficient as it could be.
*
*/
#include <chrono>
#include <iostream>
#include <set>
#include <string>
std::multiset<std::string> names;
// global data structure
void log(const std::chrono::system_clock::time_point& t, const char* s){
std::cout << "Making log entry" << std::endl;
}
void logAndAdd(const std::string& name){
auto now =
// get current time
std::chrono::system_clock::now();
log(now, "logAndAdd");
// make log entry
names.emplace(name);
// add name to global data
}
// structure; see Item 42
// for info on emplace
int main(){
std::string petName("Darla");
logAndAdd(petName);
// pass lvalue std::string
logAndAdd(std::string("Persephone"));
// pass rvalue std::string
logAndAdd("Patty Dog");
// pass string literal
}
/*
* Key idea:
*
* The inefficiencies in the second and third calls can be eliminated by
* rewriting logAndAdd to take a universal reference (see Item 24) and, in
* accord with Item 25, std::forwarding this reference to emplace.
*/
#include <chrono>
#include <iostream>
#include <set>
#include <string>
std::multiset<std::string> names;
// global data structure
void log(const std::chrono::system_clock::time_point& t, const char* s){
std::cout << "Making log entry" << std::endl;
}
template<typename T>
void logAndAdd(T&& name){
auto now = std::chrono::system_clock::now();
log(now, "logAndAdd");
names.emplace(std::forward<T>(name));
}
int main(){
std::string petName("Darla");
// as before
logAndAdd(petName);
// as before, copy
// lvalue into multiset
logAndAdd(std::string("Persephone"));
// move rvalue instead
// of copying it
logAndAdd("Patty Dog");
// create std::string
// in multiset instead of
// copying a temporary
// std::string
}
/*
* Key ideas:
*
* logAndAdd is overloaded here. Resolution of calls to the two overloads
* does not always work as expected. The overload taking a universal
* reference is invoked in the last call, and that is probably not what we
* wanted...
*
* Functions taking universal references are the greediest functions in C++.
* They instantiate to create exact matches for almost any type of argument.
* (The few kinds of arguments where this isn't the case are described in
* Item 30.) This is why combining overloading and universal references is
* almost always a bad idea: the universal reference overload vacuums up far
* more argument types than the developer doing the overloading generally
* expects.
*/
#include <chrono>
#include <iostream>
#include <set>
#include <string>
std::multiset<std::string> names;
// global data structure
void log(const std::chrono::system_clock::time_point& t, const char* s){
std::cout << "Making log entry" << std::endl;
}
std::string nameFromIdx(int idx)
// return name{
// corresponding to idx
std::string s("Test");
return s;
}
template<typename T>
void logAndAdd(T&& name){
auto now = std::chrono::system_clock::now();
log(now, "logAndAdd");
names.emplace(std::forward<T>(name));
}
void logAndAdd(int idx){
auto now = std::chrono::system_clock::now();
log(now, "logAndAdd");
names.emplace(nameFromIdx(idx));
}
int main(){
short nameIdx;
std::string petName("Darla");
// as before
logAndAdd(petName);
// as before, these
logAndAdd(std::string("Persephone"));
// calls all invoke
logAndAdd("Patty Dog");
// the T&& overload
logAndAdd(22);
// calls int overload
nameIdx = 1;
// give nameIdx a value
//logAndAdd(nameIdx); // error!
}
/*
* Key Idea:
*
* An alternative to overloading on universal references is to replace pass by
* reference with pass by value. This adheres to the guidance in Item 41 to
* consider passing objects by value when you know you'll copy them..
*/
#include <string>
#include <utility>
std::string nameFromIdx(int idx) {
return std::string("Bart");
}
class Person {
public:
explicit Person(std::string n)
// replaces T&& ctor; see
: name(std::move(n)) {}
// Item 41 for use of std::move
explicit Person(int idx)
// as before
: name(nameFromIdx(idx)) {}
// ...
private:
std::string name;
};
/*
* Key idea:
*
* TODO
*/
#include <type_traits>
#include <utility>
class Person {
public:
template<
typename T,
typename = typename std::enable_if<
!std::is_same<Person,
typename std::decay<T>::type
>::value
>::type
>
explicit Person(T&& n);
};
class SpecialPerson: public Person {
public:
SpecialPerson(const SpecialPerson& rhs)
// copy ctor; calls
: Person(rhs)
// base class{
/* ... */ } // forwarding ctor!
SpecialPerson(SpecialPerson&& rhs)
// move ctor; calls
: Person(std::move(rhs))
// base class{
/* ... */ } // forwarding ctor!
};
/*
* Key idea:
*
* TODO
*/
#include <type_traits>
class Person {
public:
template<
typename T,
typename = typename std::enable_if<
!std::is_base_of<Person,
typename std::decay<T>::type
>::value
>::type
>
explicit Person(T&& n);
// ...
};
/*
* Key idea:
*
* If we're using C++14, we can employ alias templates for std::enable_if and
* std::decay to get rid of the "typename" and "::type" cruft.
*/
#include <type_traits>
class Person {
// C++14
public:
template<
typename T,
typename = std::enable_if_t<
// less code here
!std::is_base_of<Person,
std::decay_t<T>
// and here
>::value
>
// and here
>
explicit Person(T&& n);
// ...
};
/*
* Key idea:
*
* TODO
*/
#include <string>
#include <type_traits>
std::string nameFromIdx(int idx)
// as in Item 26{
return std::string("Bart");
}
class Person {
public:
template<
typename T,
typename = std::enable_if_t<
!std::is_base_of<Person, std::decay_t<T>>::value
&&
!std::is_integral<std::remove_reference_t<T>>::value
>
>
explicit Person(T&& n)
// ctor for std::strings and
: name(std::forward<T>(n))
// args convertible to{
/* ... */ } // std::strings
explicit Person(int idx)
// ctor for integral args
: name(nameFromIdx(idx)){
/* ... */ }
// ... // copy and move ctors, etc.
private:
std::string name;
};
/*
* Key idea:
*
* TODO
*/
#include <string>
#include <type_traits>
std::string nameFromIdx(int idx)
// as in Item 26{
return std::string("Bart");
}
class Person {
public:
template<
// as before
typename T,
typename = std::enable_if_t<
!std::is_base_of<Person, std::decay_t<T>>::value
&&
!std::is_integral<std::remove_reference_t<T>>::value
>
>
explicit Person(T&& n)
: name(std::forward<T>(n)){
// assert that a std::string can be created from a T object
static_assert(
std::is_constructible<std::string, T>::value,
"Parameter n can't be used to construct a std::string"
);
// ... // the usual ctor work goes here
}
explicit Person(int idx)
// remainder of Person class (as before)
: name(nameFromIdx(idx)){
/* ... */ }
// ...
private:
std::string name;
};
/*
* Key idea:
*
* TODO
*
*/
#include <chrono>
#include <iostream>
#include <set>
std::multiset<std::string> names;
// global data structure
void log(const std::chrono::system_clock::time_point& t, const char* s){
std::cout << "Making log entry" << std::endl;
}
template<typename T>
// make log entry and add
void logAndAdd(T&& name)
// name to data structure{
auto now = std::chrono::system_clock::now();
log(now, "logAndAdd");
names.emplace(std::forward<T>(name));
}
/*
* Key idea:
*
* Tag dispatching permits us to combine universal references and overloading.
*
*/
#include <chrono>
#include <iostream>
#include <set>
#include <utility>
#include <type_traits>
std::multiset<std::string> names;
// global data structure
std::string nameFromIdx(int idx)
// as in Item 26{
return std::string("Bart");
}
void log(const std::chrono::system_clock::time_point& t, const char* s){
std::cout << "Making log entry" << std::endl;
}
template<typename T>
// non-integral
void logAndAddImpl(T&& name, std::false_type)
// argument:{
// add it to
auto now = std::chrono::system_clock::now();
// global data
log(now, "logAndAdd");
// structure
names.emplace(std::forward<T>(name));
}
template<typename T>
void logAndAdd(T&& name){
// This is not quite correct and will fail if T is an lvalue ref.
//logAndAddImpl(std::forward<T>(name),
// std::is_integral<T>()); // not quite correct
// This is correct, because we remove references.
logAndAddImpl(
std::forward<T>(name),
std::is_integral<typename std::remove_reference<T>::type>()
);
}
void logAndAddImpl(int idx, std::true_type)
// integral{
// argument: look
logAndAdd(nameFromIdx(idx));
// up name and
}
// call logAndAdd
// with it
/*
* Key ideas:
*
* - When an lvalue is passed as an argument to func, T is deduced to be an
* lvalue reference. When an rvalue is passed, T is deduced to be a
* non-reference.
*
* - Reference collapsing occurs in four contexts. The first and most common
* is template instantiation. The second is type generation for auto
* variables.
*/
#include "reference_collapsing_contexts01.h"
int main(){
Widget w;
// a variable (an lvalue)
func(w);
// call func with lvalue; T deduced
// to be Widget&
func(widgetFactory());
// call func with rvalue; T deduced
// to be Widget
auto&& w1 = w;
auto&& w2 = widgetFactory();
}
/*
* Key idea:
*
* The third context in which reference collapsing occurs is the generation
* and use of typedefs and alias declarations. If, during creation or
* evaluation of a typedef, references to references arise, reference
* collapsing intervenes to eliminate them.
*/
#include "reference_collapsing_contexts02.h"
int main(){
Widget<int&> w;
}
/*
* Key idea:
*
* References to references are illegal in C++.
*/
int main(){
int x;
//auto& & rx = x; // error! can't declare reference to reference
}
/*
* Key idea:
*
* Even types with explicit move support may not benefit as much as you'd
* hope. All containers in the standard C++11 library support moving, for
* example, but it would be a mistake to assume that moving all containers is
* cheap. For some containers, this is because there's no truly cheap way to
* move their contents. For others, it's because the truly cheap move
* operations the containers offer come with caveats the container elements
* can't satisfy.
*
* TODO: time this.
*/
#include <array>
#include <vector>
class Widget {};
int main(){
std::vector<Widget> vw1;
// ... // put data into vw1
auto vw2 = std::move(vw1);
// move vw1 into vw2. Runs in
// constant time. Only ptrs
// in vw1 and vw2 are modified
std::array<Widget, 10000> aw1;
// ... // put data into aw1
auto aw2 = std::move(aw1);
// move aw1 into aw2. Runs in
// linear time. All elements in
// aw1 are moved into aw2.
}
/*
* Key idea:
*
* A failure case for perfect forwarding is when a bitfield is used as a
* function argument.
*
* The key to passing a bitfield into a perfect-forwarding function, is to
* take advantage of the fact that the forwarded-to function will always
* receive a copy of the bitfield's value. You can thus make a copy yourself
* and call the forwarding function with the copy.
*/
#include <cstdint>
#include <utility>
struct IPv4Header {
std::uint32_t version:4,
IHL:4,
DSCP:6,
ECN:2,
totalLength:16;
// ...
};
void f(std::size_t sz) {}
// function to call
template<typename T>
void fwd(T&& param)
// accept any argument{
f(std::forward<T>(param));
// forward it to f
}
int main(){
IPv4Header h;
// ...
f(h.totalLength);
// fine
//fwd(h.totalLength); // error!
// copy bitfield value; see Item 6 for info on init. form
auto length = static_cast<std::uint16_t>(h.totalLength);
fwd(length);
// forward the copy
}
/*
* Key idea:
*
* The use of a braced initializer is a perfect forwarding failure case.
* However, there is a workaround by declaring a local variable using auto,
* and then passing the local variable to the forwarding function.
*/
#include <iostream>
#include <utility>
#include <vector>
void f(const std::vector<int>& v) {
std::cout << "f(const std::vector<int>&)" << std::endl;
}
template<typename T>
void fwd(T&& param)
// accept any argument{
f(std::forward<T>(param));
// forward it to f
}
template<typename... Ts>
void fwd(Ts&&... param)
// accept any arguments{
f(std::forward<Ts>(param)...);
// forward them to f
}
int main(){
f({1, 2, 3});
// fine, "{1, 2, 3}" implicitly
// converted to std::vector<int>
//fwd({1, 2, 3}); // error! doesn't compile
auto il = {1, 2, 3};
// il's type deduced to be
// std::initializer_list<int>
fwd(il);
// fine, perfect-forwards il to f
}
/*
* Key idea:
*
*/
#include <cstddef>
#include <iostream>
#include <vector>
void f(std::size_t val) {
std::cout << "f(std::size_t val)" << std::endl;
}
template<typename T>
void fwd(T&& param)
// accept any argument{
f(std::forward<T>(param));
// forward it to f
}
class Widget {
public:
static const std::size_t MinVals = 28;
// MinVals' declaration
// ...
};
const std::size_t Widget::MinVals;
// in Widget's .cpp file
// ... // no defn. for MinVals
int main(){
std::vector<int> widgetData;
widgetData.reserve(Widget::MinVals);
// use of MinVals
f(Widget::MinVals);
// fine, treated as "f(28)"
//fwd(Widget::MinVals); // error! shouldn't link
}
/*
* Key idea:
*
* Neither 0 or NULL can be perfect-forwarded as a null pointer. The fix is
* easy, however: pass nullptr instead of 0 or NULL
*/
#include <iostream>
#include <memory>
#include <utility>
class Widget {};
//void f(Widget *pw) {
// std::cout << "f(Widget *)" << std::endl;
//}
void f(std::unique_ptr<Widget> pw) {
std::cout << "f(std::unique_ptr<Widget>)" << std::endl;
}
template<typename T>
void fwd(T&& param)
// accept any argument{
f(std::forward<T>(param));
// forward it to f
}
template<typename... Ts>
void fwd(Ts&&... param)
// accept any arguments{
f(std::forward<Ts>(param)...);
// forward them to f
}
int main(){
f(0);
// fine, passes null pointer
f(NULL);
// also fine, also passes null pointer
f(nullptr);
//fwd(0); // error!
//fwd(NULL); // also error!
fwd(nullptr);
}
/*
* Key idea:
*
* - Overloaded function names and template names are a perfect forwarding
* failure case.
*
* - The way to get a perfect-forwarding function like fwd to accept an
* overloaded function name is to manually specify the overload or
* instantiation you want to have forwarded.
*/
#include <utility>
//void f(int (*pf)(int)); // pf = "processing function"
void f(int pf(int)) {}
// declares same f as above
int processVal(int value) { return 1; }
int processVal(int value, int priority) { return 1; }
template<typename T>
T workOnVal(T param)
// template for processing values{
T x;
return x;
}
template<typename T>
void fwd(T&& param)
// accept any argument{
f(std::forward<T>(param));
// forward it to f
}
int main(){
f(processVal);
// fine
//fwd(processVal); // error! which processVal?
//fwd(workOnVal); // error! which workOnVal
// instantiation?
using ProcessFuncType =
// make typedef;
int (*)(int);
// see Item 9
ProcessFuncType processValPtr = processVal;
// specify needed
// signature for
// processVal
fwd(processValPtr);
// fine
fwd(static_cast<ProcessFuncType>(workOnVal));
// also fine
}
#include "Widget.h"
#include "utils.h"
void Widget::addFilter() const{
// default by-value capture
filters.emplace_back(
[=](int value) { return value % divisor == 0; }
);
//filters.emplace_back( // error!
// [](int value) { return value % divisor == 0; } // divisor
//); // not
// // available
//filters.emplace_back(
// [divisor](int value) // error! no local
// { return value % divisor == 0; } // divisor to capture
//);
/* version 1: compilers treat the version with a default by-value capture like
this: */
auto currentObjectPtr = this;
filters.emplace_back(
[currentObjectPtr](int value)
{ return value % currentObjectPtr->divisor == 0; }
);
/* version 1 end */
/* version 2: make a local copy of the data member you want to capture */
auto divisorCopy = divisor;
// copy data member
// variant 1: capture the local copy
filters.emplace_back(
[divisorCopy](int value)
// capture the copy
{ return value % divisorCopy == 0; }
// use the copy
);
// variant 2: default by-value capture
filters.emplace_back(
[=](int value)
// capture the copy
{ return value % divisorCopy == 0; }
// use the copy
);
// variant 3: use generalized lambda captures
filters.emplace_back(
// C++14 only:
[divisor = divisor](int value)
// copy divisor to closure
{ return value % divisor == 0; }
// use the copy
);
/* version 2 end */
}
/*
* Key ideas: TODO
*
* TODO: fix compile error
*/
#include "Widget.h"
#include "utils.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
int computeSomeValue1(){
return 1;
}
int computeSomeValue2(){
return 2;
}
int computeDivisor(int calc1, int calc2){
return calc1 + calc2;
}
void addDivisorFilter(){
auto calc1 = computeSomeValue1();
auto calc2 = computeSomeValue2();
auto divisor = computeDivisor(calc1, calc2);
filters.emplace_back(
// danger!
[&](int value) { return value % divisor == 0; }
// ref to
// divisor
);
// will
// dangle!
filters.emplace_back(
[&divisor](int value)
// danger! ref to
{ return value % divisor == 0; }
// divisor will
);
// still dangle
filters.emplace_back(
// now
[=](int value) { return value % divisor == 0; }
// divisor
);
// can't
// dangle
}
void addDivisorFilter2(){
static auto calc1 = computeSomeValue1();
// now static
static auto calc2 = computeSomeValue2();
// now static
static auto divisor =
// now static
computeDivisor(calc1, calc2);
filters.emplace_back(
[=](int value)
// captures nothing!
{ return value % divisor == 0; }
// refers to above static
);
++divisor;
// modify divisor
}
template<typename C>
void workWithContainer(const C& container){
auto calc1 = computeSomeValue1();
// as above
auto calc2 = computeSomeValue2();
// as above
auto divisor = computeDivisor(calc1, calc2);
// as above
using ContElemT = typename C::value_type;
// type of
// elements in
// container
using std::begin;
// for
using std::end;
// genericity;
// see Item 13
// C++11 version
if (std::all_of(
// if all values
begin(container), end(container),
// in container
[&](const ContElemT& value)
// are multiple
{ return value % divisor == 0; })
// of divisor...
) {
// ... // they are...
} else {
// ... // at least one
}
// isn't...
// C++14 version:
if (std::all_of(begin(container), end(container),
[&](const auto& value)
// C++14
{ return value % divisor == 0; }))
{ } else { }
}
void doSomeWork(){
auto pw =
// create Widget; see
std::make_unique<Widget>();
// Item 21 for
// std::make_unique
pw->addFilter();
// add filter that uses
// Widget::divisor
}
// destroy Widget; filters
// now holds dangling pointer!
int main(){
filters.emplace_back(
// see Item 42 for
[](int value) { return value % 5 == 0; }
// info on
);
// emplace_back
}
#include "utils.h"
FilterContainer filters;
// filtering funcs
#include "Widget.h"
bool Widget::isValidated() const { return true; }
bool Widget::isProcessed() const { return true; }
bool Widget::isArchived() const { return true; }
/**
* Key idea:
*
* Example showing how you can accomplish move capture in a language lacking
* support for move capture.
*
* A lambda expression is simply a way to cause a class to be generated and an
* object of that type to be created. There is nothing you can do with a
* lambda that you can't do by hand.
*/
#include <memory>
#include "Widget.h"
class IsValAndArch {
// "is validated
public:
// and archived"
using DataType = std::unique_ptr<Widget>;
explicit IsValAndArch(DataType&& ptr)
// Item 25 explains
: pw(std::move(ptr)) {}
// use of std::move
bool operator()() const
{ return pw->isValidated() && pw->isArchived(); }
private:
DataType pw;
};
int main(){
// REMARK: due to the use of std::make_unique, this line is C++14 and not
// C++11!!!
auto func = IsValAndArch(std::make_unique<Widget>());
}
/**
* Key idea:
*
* C++14 example showing how one can use init capture to move a
* std::unique_ptr into a closure.
*/
#include <memory>
#include <utility>
#include "Widget.h"
int main(){
auto pw = std::make_unique<Widget>();
// create Widget; see
// Item 21 for info on
// std::make_unique
// configure *pw{
auto func = [pw = std::move(pw)]
// init data mbr
{ return pw->isValidated()
// in closure w/
&& pw->isArchived(); };
// std::move(pw)
}{
auto func = [pw = std::make_unique<Widget>()]
// init data mem
{ return pw->isValidated()
// in closure w/
&& pw->isArchived(); };
// result of call
// to make_unique
}
}
/**
* Key idea:
*
* C++11 example showing how you can emulate move capture in C++11 by
*
* 1. moving the object to be captured into a function object produced by
* std::bind and
* 2. giving the lambda a reference to the "captured" object.
*/
#include <functional>
#include <utility>
#include <vector>
int main(){
std::vector<double> data;
// object to be moved
// into closure
// ... // populate data{
auto func =
std::bind(
// C++11 emulation
[](const std::vector<double>& data)
// of init capture{
/* uses of data */ },
std::move(data)
);
}{
auto func =
std::bind(
// C++11 emulation
[](std::vector<double>& data) mutable
// of init capture{
/* uses of data */ }, // for mutable lambda
std::move(data)
);
}
}
/**
* Key idea:
*
* C++14 example showing how you can create a local std::vector, put an
* appropriate set of values into it, and then move it into a closure.
*/
#include <utility>
#include <vector>
int main(){
std::vector<double> data;
// object to be moved
// into closure
// ... // populate data
auto func = [data = std::move(data)]
// C++14 init capture{
/* uses of data */ };
}
/**
* Key idea:
*
* Example demonstrating how to use std::bind to emulate move capture.
*/
#include <functional>
#include <memory>
#include <utility>
#include "Widget.h"
int main(){{
// C++14 version
auto func = [pw = std::make_unique<Widget>()]
// as before,
{ return pw->isValidated()
// create pw
&& pw->isArchived(); };
// in closure
}{
// C++11 emulation
auto func = std::bind(
[](const std::unique_ptr<Widget>& pw)
{ return pw->isValidated()
&& pw->isArchived(); },
std::make_unique<Widget>()
// REMARK: this is C++14!!!
);
}
}
/**
* Key idea:
*
* Use decltype on auto&& parameters to std::forward them.
*/
#include <utility>
int normalize(int x) { return x; }
int func(int x) { return x; }
int main(){{
// If normalize treats lvalues differently from rvalues, this lambda isn't
// really written properly, because it always passes an lvalue (the
// parameter x) to normalize, even if the argument that was passed to the
// lambda was an rvalue.
auto f = [](auto x){ return func(normalize(x)); };
}{
auto f =
[](auto&& param)
{ return func(normalize(std::forward<decltype(param)>(param))); };
}{
// C++14 lambdas can also be variadic.
auto f =
[](auto&&... params)
{ return func(normalize(std::forward<decltype(params)>(params)...)); };
}
}
#include "alarm.h"
#include "alarm_extra.h"
auto setSoundL =
[](Sound s){
using namespace std::chrono;
using namespace std::literals;
// for C++14 suffixes
setAlarm(steady_clock::now() + 1h,
// fine, calls
s,
// 3-arg version
30s);
// of setAlarm
};
int main(){
setSoundL(Sound::Siren);
// body of setAlarm may
// well be inlined here
}
#include "alarm.h"
// setSoundL ("L" for "lambda") is a function object allowing a
// sound to be specified for a 30-sec alarm to go off an hour
// after it's set
auto setSoundL =
[](Sound s){
// make std::chrono components available w/o qualification
using namespace std::chrono;
setAlarm(steady_clock::now() + hours(1),
// alarm to go off
s,
// in an hour for
seconds(30));
// 30 seconds
};
int main(){
setSoundL(Sound::Siren);
// body of setAlarm may
// well be inlined here
}
#include "alarm.h"
auto setSoundL =
[](Sound s){
using namespace std::chrono;
using namespace std::literals;
// for C++14 suffixes
setAlarm(steady_clock::now() + 1h,
// C++14, but // fine, calls
s,
// same meaning // 3-arg version
30s);
// as above // of setAlarm
};
int main(){
setSoundL(Sound::Siren);
// body of setAlarm may
// well be inlined here
}
#include "alarm.h"
#include "alarm_extra.h"
using namespace std::chrono;
// as above
using namespace std::literals;
using namespace std::placeholders;
// needed for use of "_1"
//auto setSoundB = // error! which
// std::bind(setAlarm, // setAlarm?
// std::bind(std::plus<>(),
// steady_clock::now(),
// 1h),
// _1,
// 30s);
using SetAlarm3ParamType = void(*)(Time t, Sound s, Duration d);
auto setSoundB =
// now
std::bind(static_cast<SetAlarm3ParamType>(setAlarm),
// okay
std::bind(std::plus<>(),
steady_clock::now(),
1h),
_1,
30s);
int main(){
setSoundB(Sound::Siren);
// body of setAlarm is less
// likely to be inlined here
}
#include "alarm.h"
using namespace std::chrono;
// as above
using namespace std::placeholders;
auto setSoundB =
std::bind(setAlarm,
std::bind(std::plus<steady_clock::time_point>(), steady_clock::now(), hours(1)),
_1,
seconds(30));
int main(){
setSoundB(Sound::Siren);
// body of setAlarm is less
// likely to be inlined here
}
#include "alarm.h"
using namespace std::chrono;
// as above
using namespace std::literals;
using namespace std::placeholders;
// needed for use of "_1"
auto setSoundB =
// "B" for "bind"
std::bind(setAlarm,
std::bind(std::plus<>(), steady_clock::now(), 1h),
_1,
30s);
int main(){
setSoundB(Sound::Siren);
// body of setAlarm is less
// likely to be inlined here
}
#include "alarm.h"
using namespace std::chrono;
// as above
using namespace std::literals;
using namespace std::placeholders;
// needed for use of "_1"
auto setSoundB =
// "B" for "bind"
std::bind(setAlarm,
steady_clock::now() + 1h,
// incorrect! see below
_1,
30s);
auto betweenL =
// C++11 version
[lowVal, highVal]
(int val)
{ return lowVal <= val && val <= highVal; };
auto betweenL =
[lowVal, highVal]
(const auto& val)
// C++14
{ return lowVal <= val && val <= highVal; };
auto betweenB =
// C++11 version
std::bind(std::logical_and<bool>(),
std::bind(std::less_equal<int>(), lowVal, _1),
std::bind(std::less_equal<int>(), _1, highVal));
using namespace std::placeholders;
// as above
auto betweenB =
std::bind(std::logical_and<>(),
// C++14
std::bind(std::less_equal<>(), lowVal, _1),
std::bind(std::less_equal<>(), _1, highVal));
#include "compress.h"
auto compressRateL =
// w is captured by
[w](CompLevel lev)
// value; lev is
{ return compress(w, lev); }
// passed by value
int main(){
compressRateL(CompLevel::High);
// arg is passed
// by value
}
#include "compress.h"
Widget w;
using namespace std::placeholders;
auto compressRateB = std::bind(compress, w, _1);
int main(){
compressRateB(CompLevel::High);
// how is arg
// passed?
}
class PolyWidget {
public:
template<typename T>
void operator()(const T& param);
}
PolyWidget pw;
auto boundPW = [pw](const auto& param)
// C++14
{ pw(param); };
int main(){
boundPW(1930);
// pass int to
// PolyWidget::operator()
boundPW(nullptr);
// pass nullptr to
// PolyWidget::operator()
boundPW("Rosebud");
// pass string literal to
// PolyWidget::operator()
}
class PolyWidget {
public:
template<typename T>
void operator()(const T& param);
}
PolyWidget pw;
auto boundPW = std::bind(pw, _1);
int main(){
boundPW(1930);
// pass int to
// PolyWidget::operator()
boundPW(nullptr);
// pass nullptr to
// PolyWidget::operator()
boundPW("Rosebud");
// pass string literal to
// PolyWidget::operator()
}
/**
* Key idea:
*
* - Software threads are a limited resource. If you try to create more than the system
* can provide, a std::system_error exception is thrown. This is true even if the function
* you want to run can't throw.
*/
#include <iostream>
#include <thread>
#include <vector>
int doAsyncWork() noexcept
// see Item 14 for noexcept{
std::cout << "doAsyncWork()" << std::endl;
return 1;
}
int main(){
std::vector<std::thread> threads;
unsigned int nr_threads = 10;
// increase this until you get a std::system_error exception...???
for (unsigned int i = 0; i < nr_threads; ++i){
// TODO: why doesn't this work???
//std::thread t(doAsyncWork); // throws if no more
// // threads are available
//threads.push_back(t);
threads.push_back(std::thread(doAsyncWork));
}
for (auto& t : threads){
t.join();
}
return 0;
}
/**
* Key ideas:
*
* - The task-based approach is typically superior to its thread-based counterpart.
*
* - With the task-based approach, it is easy to get access to the return value of doAsyncWork,
* because the future returned from std::async offers the get function.
*
* - The get function also provides access to the exception if doAsyncWork throws an exception.
*/
#include <future>
#include <iostream>
int doAsyncWork(){
std::cout << "doAsyncWork()" << std::endl;
//throw;
return 1;
}
int main(){
auto fut = std::async(doAsyncWork);
// onus of thread mgmt is
// on implementer of
// the Standard Library
int ret = fut.get();
std::cout << "doAsyncWork() returned " << ret << std::endl;
return 0;
}
/**
* Key ideas:
*
* - The task-based approach is typically superior to its thread-based counterpart.
*
* - With the thread-based invocation, there's no straightforward way to get access
* to the return value of doAsyncWork.
*
* - With the thread-based approach, if doAsyncWork throws, the program dies (via a call
* to std::terminate).
*/
#include <iostream>
#include <thread>
int doAsyncWork(){
std::cout << "doAsyncWork()" << std::endl;
//throw;
return 1;
}
int main(){
std::thread t(doAsyncWork);
t.join();
return 0;
}
/**
* Key idea:
*
* std::async's default launch policy is std::launch::async or-ed together with
* std::launch::deferred.
*/
#include <future>
#include <iostream>
void f(){
std::cout << "f()" << std::endl;
}
int main(){
auto fut1 = std::async(f);
// run f using
// default launch
// policy
auto fut2 = std::async(std::launch::async |
// run f either
std::launch::deferred,
// async or
f);
// deferred
fut1.get();
fut2.get();
return 0;
}
/**
* Key idea:
*
* The way to guarantee that std::async will schedule the task for truly
* asynchronous execution is to pass std::launch::async as the first argument
* when you make the call to std::async.
*/
#include <future>
#include <iostream>
void f(){
std::cout << "f()" << std::endl;
}
int main(){
auto fut = std::async(std::launch::async, f);
// launch f
// asynchronously
fut.get();
return 0;
}
/**
* Key idea:
*
* Using std::async with the default launch policy has some interesting implications.
*
* TODO: improve this example.
*/
#include <future>
#include <iostream>
void f(){
std::cout << "f()" << std::endl;
}
int main(){
auto fut = std::async(f);
// run f using default launch policy
}
/**
* Key idea:
*
* This is a C++11 version of a function that acts like std::async, but that
* automatically uses std::launch::async as the launch policy.
*/
#include <future>
#include <iostream>
template<typename F, typename... Ts>
inline
std::future<typename std::result_of<F(Ts...)>::type>
reallyAsync(F&& f, Ts&&... params)
// return future{
// for asynchronous
return std::async(std::launch::async,
// call to f(params...)
std::forward<F>(f),
std::forward<Ts>(params)...);
}
void f(){
std::cout << "f()" << std::endl;
}
int main(){
auto fut = reallyAsync(f);
// run f asynchronously;
// throw if std::async
// would throw
}
/**
* Key idea:
*
* This is a C++14 version of a function that acts like std::async, but that
* automatically uses std::launch::async as the launch policy.
*/
#include <future>
#include <iostream>
template<typename F, typename... Ts>
inline
auto
// C++14
reallyAsync(F&& f, Ts&&... params){
return std::async(std::launch::async,
std::forward<F>(f),
std::forward<Ts>(params)...);
}
void f(){
std::cout << "f()" << std::endl;
}
int main(){
auto fut = reallyAsync(f);
// run f asynchronously;
// throw if std::async
// would throw
}
/**
* Key idea:
*
* The default launch policy's scheduling flexibility often mixes poorly with the use
* of thread_local variables, because it means that if f reads or writes such thread-local
* storage (TLS), it's not possible to predict which thread's variables will be accessed.
*
* TODO: improve this example.
*/
#include <future>
#include <iostream>
void f(){
std::cout << "f()" << std::endl;
}
int main(){
auto fut = std::async(f);
// TLS for f possibly for
// independent thread, but
// possibly for thread
// invoking get or wait on fut
}
/** Key idea:
*
* - The default launch policy's scheduling flexibility also affects
* wait-based for loops using timeouts, because calling wait_for or
* wait_until on a task (see Item 35) that's deferred yields the value
* std::launch::deferred.
*/
#include <future>
using namespace std::literals;
// for C++14 duration
// suffixes; see Item 34
void f()
// f sleeps for 1 second,{
// then returns
std::this_thread::sleep_for(1s);
}
int main(){
auto fut = std::async(f);
// run f asynchronously
// (conceptually)
while (fut.wait_for(100ms) !=
// loop until f has
std::future_status::ready)
// finished running...{
// which may never happen!
// ...
}
}
/** Key idea:
*
* TODO
*
*/
#include <future>
using namespace std::literals;
void f(){
std::this_thread::sleep_for(1s);
}
int main(){
auto fut = std::async(f);
// as above
if (fut.wait_for(0s) ==
// if task is
std::future_status::deferred)
// deferred...{
// ...use wait or get on fut
// to call f synchronously
} else {
// task isn't deferred
while (fut.wait_for(100ms) !=
// infinite loop not
std::future_status::ready) {
// possible (assuming
// f finishes)
// task is neither deferred nor ready,
// so do concurrent work until it's ready
}
// ... // fut is ready
}
}
/*
* If conditionsAreSatisfied() returns true, all is well, but if it returns
* false or throws an exception, the std::thread object t will be joinable when
* its destructor is called at the end of doWork. That would cause program
* execution to be terminated.
*/
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include "utils.h"
#if __cplusplus == 201103L
constexpr auto tenMillion = 10000000;
// see Item 15
// for constexpr
#elif __cplusplus == 201402L
constexpr auto tenMillion = 10'000'000;
// C++14
#endif
bool doWork(std::function<bool(int)> filter,
// returns whether
int maxVal = tenMillion)
// computation was{
// performed; see
// Item 2 for
// std::function
std::vector<int> goodVals;
// values that
// satisfy filter
std::thread t([&filter, maxVal, &goodVals]
// populate{
// goodVals
for (auto i = 0; i <= maxVal; ++i)
{ if (filter(i)) goodVals.push_back(i); }
});
auto nh = t.native_handle();
// use t's native
// ... // handle to set
// t's priority
if (conditionsAreSatisfied()) {
t.join();
// let t finish
performComputation(goodVals);
return true;
// computation was
}
// performed
return false;
// computation was
}
// not performed
int main(){
std::function<bool(int)> filter = someFilter;
doWork(filter, 10);
}
#include "ThreadRAII.h"
#include "utils.h"
#include <functional>
#if __cplusplus == 201103L
constexpr auto tenMillion = 10000000;
// see Item 15
// for constexpr
#elif __cplusplus == 201402L
constexpr auto tenMillion = 10'000'000;
// C++14
#endif
bool doWork(std::function<bool(int)> filter,
// as before
int maxVal = tenMillion){
std::vector<int> goodVals;
// as before
ThreadRAII t(
std::thread([&filter, maxVal, &goodVals]{
for (auto i = 0; i <= maxVal; ++i)
{ if (filter(i)) goodVals.push_back(i); }
}),
ThreadRAII::DtorAction::join
// RAII action
);
auto nh = t.get().native_handle();
// ...
if (conditionsAreSatisfied()) {
t.get().join();
performComputation(goodVals);
return true;
}
return false;
}
int main(){
std::function<bool(int)> filter = someFilter;
doWork(filter, 10);
}
#include <future>
#include <vector>
// this container *might* block in its dtor, because one or more
// contained futures could refer to a shared state for a non-
// deferred task launched via std::async
std::vector<std::future<void>> futs;
// see Item 39 for info
// on std::future<void>
class Widget {
// Widget objects *might*
public:
// block in their dtors
// ...
private:
std::shared_future<double> fut;
};
#include <future>
#include <thread>
#include <utility>
int calcValue() { return 1; };
// func to run
int main(){
// begin block
std::packaged_task<int()>
// wrap calcValue so it
pt(calcValue);
// can run asynchronously
auto fut = pt.get_future();
// get future for pt
std::thread t(std::move(pt));
// run pt on t
// ... // see below
}
// end block
