/**
* ============================================================================
* File: Builder_Stack.cpp (Pure Stack Version)
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This version is optimized for maximum performance by avoiding the Heap
* entirely. All components, including the collection of wheels, are
* allocated on the Stack.
*
* --- STACK ALLOCATION & EFFICIENCY:
* Unlike the dynamic version, this implementation returns the final object
* by value. This avoids the overhead of heap allocation (new/delete) and
* is ideal for objects whose lifetime is tied to the local scope.
*
* --- STACK POLYMORPHISM:
* Instead of 'std::unique_ptr<Wheel>', we use 'std::variant'. This allows
* different wheel types to coexist in a 'std::array' without dynamic
* allocation or pointer indirection.
*
* --- ADVANCED C++ FEATURES:
* 1. Fluent Interface: Setter methods return a reference to the builder 
*    (*this), allowing for elegant method chaining.
* 2. Conversion Operator: The implementation includes 'operator Car()', 
*    enabling implicit conversion from the Builder to the Product.
* 3. Explicit Build: A '.build()' method is provided for cases where 
*    explicit object finalization is preferred for clarity.
* 4. Stack Polymorphism: Uses 'std::variant' and 'std::array' to handle 
*    different wheel types without heap-based pointers.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <array>
#include <variant>
// -------------------------- Components of a Car:
class Engine
{
private:
int power_;
public:
explicit Engine(int power) : power_{power} { }
int get_power() const { return power_; }
};
// No need for a base class with std::variant
class StandardWheel  {
/*...*/ };
class HeavyDutyWheel {
/*...*/ };
// Variant acts as a stack-based polymorphic container
using WheelVariant = std::variant<StandardWheel, HeavyDutyWheel>;
//--------------------------------------- The Car:
class Car
{
public:
enum class Type {Family, Truck, Sport};
// --- THE RULE OF SEVEN (MGQ Mnemonic System) ---
Car()                          = delete;
// 1 DC: No default car to force the use of the Builder
Car(const Car&)                = default;
// 2 CC: Possible on stack
Car(Car&&) noexcept            = default;
// 3 MC: Efficient transfer
Car& operator=(const Car&)     = default;
// 4 CA: Possible on stack
Car& operator=(Car&&) noexcept = default;
// 5 MA: Efficient re-assignment
~Car()                         = default;
// 6 De: Destructor
private:
// Static constraint: Max wheels on the stack
static constexpr int MAX_WHEELS = 8;
float         weight_;
float         length_;
float         width_;
int           doorCount_;
std::string   color_;
Type          type_;
Engine        engine_;
int           activeWheels_;
// STACK STORAGE: Fixed size array of variants for wheels
using Wheels_array = std::array<WheelVariant, MAX_WHEELS>;
Wheels_array wheels_;
// 7 PC: Very complicated particular constructor (intentionally private):
Car(float weight, float length, float width, int doorCount, std::string color,
Type type, Engine engine, int activeWheels, Wheels_array wheels)
: weight_{weight}, length_{length}, width_{width}, doorCount_{doorCount},
color_{std::move(color)}, type_{type}, engine_{engine}, 
activeWheels_{activeWheels}, wheels_{wheels} { }
friend class Builder;
// Only local Builder class can build Cars
public:
void print() const
{
std::cout << "Car: weight = " << weight_ << ", length = " << length_ << ", width = " << width_
<< ", doorCount = " << doorCount_ << ", wheels = " << activeWheels_ << ", color = "
<< color_ << ", enginePower = " << engine_.get_power() << ", type = "
<< (type_==Type::Family ? "Family" : type_==Type::Truck ? "Truck" : "Sport")
<< std::endl;
}
class Builder final
// Use Builder to manage such complicated constructor.
{
private:
float       weight_     {1.3f};
// default values
float       length_     {2.2f};
float       width_      {1.8f};
int         doorCount_  {4};
std::string color_      {"black"};
int         power_      {100};
int         wheelCount_ {4};
public:
Builder& setWeight     (float we)       noexcept {weight_     = we;                            return *this;}
Builder& setLength     (float le)       noexcept {length_     = le;                            return *this;}
Builder& setWidth      (float wi)       noexcept {width_      = wi;                            return *this;}
Builder& setDoorCount  (int dc)         noexcept {doorCount_  = dc;                            return *this;}
Builder& setColor      (std::string co) noexcept {color_      = std::move(co);                 return *this;}
Builder& setPower      (int po)         noexcept {power_      = po;                            return *this;}
Builder& setWheelCount (int wc)         noexcept {wheelCount_ = (wc>MAX_WHEELS)?MAX_WHEELS:wc; return *this;}
Car build() const
{
Type type;
if(power_ > 400) type = (wheelCount_ > 4) ? Type::Truck : Type::Sport;
else             type = Type::Family;
Car::Wheels_array wheels;
for(int i{0}; i < wheelCount_; ++i)
{
if(wheelCount_ > 4) wheels[i] = HeavyDutyWheel{};
else                wheels[i] = StandardWheel{};
}
return Car{weight_, length_, width_, doorCount_, std::move(color_),
type, Engine(power_), wheelCount_, wheels};
}
// Build
// Conversion Operator
operator Car() const { return build(); }
};
// Builder
};
// Car
//------------------------------------------------------------------------------- Main
int main()
{
// Example 1: Default-ish construction
Car car = Car::Builder{};
// calling .build() is not necessary due to implicit conversion.
car.print();
// Example 2: Sport configuration (Fluent interface)
car = Car::Builder{}.setColor("white")
.setDoorCount(3)
.setWidth(1.6f)
.setLength(4.0f)
.setWheelCount(4)
.setPower(550)
.build();
// anyway, .build() can be called!
car.print();
// Example 3: Truck configuration (Fluent interface)
car = Car::Builder{}.setLength(5.5f)
.setWidth(2.6f)
.setWeight(3.1f)
.setDoorCount(2)
.setWheelCount(6)
.setPower(900);
car.print();
// Demonstration of Copying
Car car2 = car;
// Calls 2 CC (Copy Constructor)
std::cout << "Car 2 (Copy of Car):\n";
car2.print();
}
//================================================================================ END
/**
* ============================================================================
* File: Builder.cpp (Dynamic Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Builder pattern in a dynamic context. 
* The main goal is to separate the construction of a complex object (Car) 
* from its internal representation. 
* 
* --- DYNAMIC ALLOCATION & SMART POINTERS:
* In this version, the Builder creates the final object in the Heap and 
* returns a 'std::unique_ptr<Car>'. This is the preferred approach when:
* 1. The object is too large for the stack.
* 2. The object contains polymorphic components (e.g., different types 
*    of Engines or Wheels) that require pointer stability.
* 3. The object's lifetime needs to be managed across different scopes.
* 
* --- FLUENT INTERFACE:
* The Builder uses method chaining (setters returning *this) to allow 
* a readable and expressive construction process in a single statement.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <vector>
// -------------------------- Components of a Car:
class Engine
{
private:
int power_;
public:
explicit Engine(int power) : power_{power} { }
int get_power() const { return power_; }
};
class Wheel {
/*...*/};
class StandardWheel  : public Wheel {
/*...*/};
class HeavyDutyWheel : public Wheel {
/*...*/};
//--------------------------------------- The Car:
class Car
{
public:
enum class Type {Family, Truck, Sport};
// --- THE RULE OF SEVEN (MGQ Mnemonic System) ---
Car()                          = delete;
// 1 DC: No default car to force the use of the Builder
Car(const Car&)                = delete;
// 2 CC: Not possible because unique_ptrs can't be copied
Car(Car&&) noexcept            = default;
// 3 MC: Possible because unique_ptrs can be moved
Car& operator=(const Car&)     = delete;
// 4 CA: Not possible because unique_ptrs can't be copied
Car& operator=(Car&&) noexcept = default;
// 5 MA: Possible because unique_ptrs can be moved
~Car()                         = default;
// 6 De: Destructor
private:
using Engine_ptr    = std::unique_ptr<Engine>;
// These are necessary because Engine and
using Wheel_ptr     = std::unique_ptr<Wheel>;
// Wheel classes could be polymorphic.
using Wheels_vector = std::vector<Wheel_ptr>;
float         weight_;
float         length_;
float         width_;
int           doorCount_;
std::string   color_;
Type          type_;
Engine_ptr    engine_;
Wheels_vector wheels_;
// 7 PC: Very complicated particular constructor (intentionally private):
Car(float weight, float length, float width, int doorCount, std::string color,
Type type, Engine_ptr engine, Wheels_vector wheels)
: weight_{weight}, length_{length}, width_{width}, doorCount_{doorCount},
color_{std::move(color)}, type_{type}, engine_{std::move(engine)},
wheels_{std::move(wheels)} { }
friend class Builder;
// Only local Builder class can build Cars
public:
void print() const
{
std::cout << "Car: weight = " << weight_ << ", length = " << length_ << ", width = " << width_
<< ", doorCount = " << doorCount_ << ", wheels = " << wheels_.size() << ", color = "
<< color_ << ", enginePower = " << engine_->get_power() << ", type = "
<< (type_==Type::Family ? "Family" : type_==Type::Truck ? "Truck" : "Sport")
<< std::endl;
}
class Builder final
// Use Builder to manage such complicated constructor.
{
private:
float       weight_     {1.3f};
// default values
float       length_     {2.2f};
float       width_      {1.8f};
int         doorCount_  {4};
std::string color_      {"black"};
int         power_      {100};
int         wheelCount_ {4};
template<class T, typename=std::enable_if_t<std::is_base_of_v<Wheel, T>>>
Wheels_vector buildWheels() const
{
Wheels_vector wheels;
wheels.reserve(wheelCount_);
for(auto i{0}; i<wheelCount_; ++i) wheels.emplace_back(std::make_unique<T>());
return wheels;
}
public:
Builder& setWeight     (float we)       noexcept {weight_     = we;            return *this;}
Builder& setLength     (float le)       noexcept {length_     = le;            return *this;}
Builder& setWidth      (float wi)       noexcept {width_      = wi;            return *this;}
Builder& setDoorCount  (int dc)         noexcept {doorCount_  = dc;            return *this;}
Builder& setColor      (std::string co) noexcept {color_      = std::move(co); return *this;}
Builder& setPower      (int po)         noexcept {power_      = po;            return *this;}
Builder& setWheelCount (int wc)         noexcept {wheelCount_ = wc;            return *this;}
std::unique_ptr<Car> build() const
{
Type type;
if(power_ > 400) type = (wheelCount_ > 4) ? Type::Truck : Type::Sport;
else             type = Type::Family;
Wheels_vector wheels;
if(wheelCount_>4) wheels = buildWheels<HeavyDutyWheel>();
else              wheels = buildWheels<StandardWheel>();
return std::unique_ptr<Car>(new Car{weight_, length_, width_, doorCount_, std::move(color_),
type, std::make_unique<Engine>(power_),
std::move(wheels)});
}
// Build
// Conversion Operator
operator std::unique_ptr<Car>() const { return build(); }
};
// Builder
};
// Car
//------------------------------------------------------------------------------- Main
int main()
{
// Example 1: Default-ish construction
std::unique_ptr<Car> car = Car::Builder{};
// calling .build() is not necessary due to implicit conversion.
car->print();
// Example 2: Sport configuration (Fluent interface)
car = Car::Builder{}.setColor("white")
.setDoorCount(3)
.setWidth(1.6f)
.setLength(4.0f)
.setWheelCount(4)
.setPower(550)
.build();
// anyway, .build() can be called!
car->print();
// Example 3: Truck configuration (Fluent interface)
car = Car::Builder{}.setLength(5.5f)
.setWidth(2.6f)
.setWeight(3.1f)
.setDoorCount(2)
.setWheelCount(6)
.setPower(900);
car->print();
}
//================================================================================= END
/**
* ============================================================================
* File: Builder_Deducing.cpp (Modern C++23 Version)
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This version evolves the Static Builder by using the C++23 feature: 
* "Deducing This" (Explicit Object Parameters) applied to a Multi-Mixin 
* architecture.
*
* --- THE ARCHITECTURAL ADVANTAGE (MULTI-MIXIN COMPOSITION):
* Traditionally, splitting a Builder into multiple specialized classes 
* (e.g., Physical, Mechanical, and Aesthetic) created a "Return Type Problem". 
* If a method in a base class returns a reference to itself, the Fluent 
* Interface breaks because the compiler "forgets" the derived Builder type.
*
* C++23 'this auto&& self' solves this elegantly. By using the explicit 
* object parameter, methods in any parent class automatically deduce and 
* return the outermost derived type (the Builder). This allows us to:
* 1. Modularize State: Each Mixin class manages its own domain of properties.
* 2. Maintain Encapsulation: Using 'class' with 'protected' data ensures 
*    that properties are only modified through the public Fluent API.
* 3. Zero Complexity: We avoid the "Template Hell" of CRTP while keeping 
*    the performance of static polymorphism.
*
* --- TECHNICAL MECHANICS:
* 1. Deducing This: 'auto&& setter(this auto&& self, ...)' captures the 
*    actual Builder instance, even when called from a base class.
* 2. Value Category Preservation: Using 'auto&&' and 'std::forward' ensures 
*    that the Builder works perfectly with both lvalues and temporaries.
* 3. Multi-Inheritance: The Builder acts as a pure orchestrator, inheriting 
*    all state and setters from its specialized parents.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <array>
#include <variant>
#include <utility>
// Required for std::forward
// -------------------------- Components of a Car:
class Engine
{
private:
int power_;
public:
explicit Engine(int power) : power_{power} { }
int get_power() const { return power_; }
};
// No need for a base class with std::variant
class StandardWheel  {
/*...*/ };
class HeavyDutyWheel {
/*...*/ };
// Variant acts as a stack-based polymorphic container
using WheelVariant = std::variant<StandardWheel, HeavyDutyWheel>;
//--------------------------------------- The Car:
class Car
{
public:
enum class Type {Family, Truck, Sport};
// --- THE RULE OF SEVEN (MGQ Mnemonic System) ---
Car()                          = delete;
// 1 DC: No default car to force the use of the Builder
Car(const Car&)                = default;
// 2 CC: Possible on stack
Car(Car&&) noexcept            = default;
// 3 MC: Efficient transfer
Car& operator=(const Car&)     = default;
// 4 CA: Possible on stack
Car& operator=(Car&&) noexcept = default;
// 5 MA: Efficient re-assignment
~Car()                         = default;
// 6 De: Destructor
private:
// Static constraint: Max wheels on the stack
static constexpr int MAX_WHEELS = 8;
float         weight_;
float         length_;
float         width_;
int           doorCount_;
std::string   color_;
Type          type_;
Engine        engine_;
int           activeWheels_;
// STACK STORAGE: Fixed size array of variants for wheels
using Wheels_array = std::array<WheelVariant, MAX_WHEELS>;
Wheels_array wheels_;
// 7 PC: Very complicated particular constructor (intentionally private):
Car(float weight, float length, float width, int doorCount, std::string color,
Type type, Engine engine, int activeWheels, Wheels_array wheels)
: weight_{weight}, length_{length}, width_{width}, doorCount_{doorCount},
color_{std::move(color)}, type_{type}, engine_{engine}, 
activeWheels_{activeWheels}, wheels_{wheels} { }
friend class Builder;
// Only local Builder class can build Cars
public:
void print() const
{
std::cout << "Car: weight = " << weight_ << ", length = " << length_ << ", width = " << width_
<< ", doorCount = " << doorCount_ << ", wheels = " << activeWheels_ << ", color = "
<< color_ << ", enginePower = " << engine_.get_power() << ", type = "
<< (type_==Type::Family ? "Family" : type_==Type::Truck ? "Truck" : "Sport")
<< std::endl;
}
// --- Hierarchical Builder Mixins using C++23 Deducing This ---
class PhysicalProps
{
protected:
float weight_ {1.3f};
float length_ {2.2f};
float width_  {1.8f};
public:
// Using auto&& self handles both lvalues and rvalues (temporaries)
auto&& setWeight(this auto&& self, float we) noexcept
{
self.weight_ = we; 
return std::forward<decltype(self)>(self);
}
auto&& setLength(this auto&& self, float le) noexcept
{
self.length_ = le; 
return std::forward<decltype(self)>(self);
}
auto&& setWidth(this auto&& self, float wi) noexcept
{
self.width_ = wi; 
return std::forward<decltype(self)>(self);
}
};
class MechanicalProps
{
protected:
int power_ {100};
public:
auto&& setPower(this auto&& self, int po) noexcept
{
self.power_ = po;
return std::forward<decltype(self)>(self);
}
};
class AestheticProps
{
protected:
int         doorCount_  {4};
std::string color_      {"black"};
int         wheelCount_ {4};
public:
auto&& setDoorCount(this auto&& self, int dc) noexcept
{
self.doorCount_ = dc;
return std::forward<decltype(self)>(self);
}
auto&& setColor(this auto&& self, std::string co) noexcept
{
self.color_ = std::move(co);
return std::forward<decltype(self)>(self);
}
auto&& setWheelCount(this auto&& self, int wc) noexcept
{
self.wheelCount_ = (wc > MAX_WHEELS) ? MAX_WHEELS : wc;
return std::forward<decltype(self)>(self);
}
};
// Final Builder composed of specialized Mixins
class Builder final : public PhysicalProps, public MechanicalProps, public AestheticProps
{
public:
Car build() const
{
Type type;
if(power_ > 400) type = (wheelCount_ > 4) ? Type::Truck : Type::Sport;
else             type = Type::Family;
Car::Wheels_array wheels;
for(int i{0}; i < wheelCount_; ++i)
{
if(wheelCount_ > 4) wheels[i] = HeavyDutyWheel{};
else                wheels[i] = StandardWheel{};
}
return Car{weight_, length_, width_, doorCount_, std::move(color_),
type, Engine(power_), wheelCount_, wheels};
}
// Build
// Conversion Operator
operator Car() const { return build(); }
};
// Builder
};
// Car
//------------------------------------------------------------------------------- Main
int main()
{
// Example 1: Default-ish construction
Car car = Car::Builder{};
// calling .build() is not necessary due to implicit conversion.
car.print();
// Example 2: Sport configuration (Fluent interface)
car = Car::Builder{}.setColor("white")
.setDoorCount(3)
.setWidth(1.6f)
.setLength(4.0f)
.setWheelCount(4)
.setPower(550)
.build();
// anyway, .build() can be called!
car.print();
// Example 3: Truck configuration (Fluent interface)
car = Car::Builder{}.setLength(5.5f)
.setWidth(2.6f)
.setWeight(3.1f)
.setDoorCount(2)
.setWheelCount(6)
.setPower(900);
car.print();
// Demonstration of Copying
Car car2 = car; 
std::cout << "Car 2 (Copy of Car):\n";
car2.print();
}
//================================================================================ END
/**
* ============================================================================
* File: SimpleFactoryMethod.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- EDUCATIONAL OVERVIEW: THE "SIMPLE FACTORY" IDIOM ---
* 
* Game Concept:
* This is a multi-level space game where meteors increase in size and speed 
* as the player advances. Whenever a meteor is spawned, the program simply 
* calls its showInfo() method to print its stats.
* 
* The Architecture (Simple / Parameterized Factory):
* In this approach, we use a SINGLE concrete factory class (MeteorFactory). 
* The client changes the factory's internal state by calling setLevel(int). 
* When asked to create a meteor, the factory uses a control structure (like a 
* 'switch' or 'if/else' statement) to decide which concrete meteor class to 
* instantiate based on the current level.
* 
* --- DEFICIENCIES & WHY THIS IS NOT THE GoF FACTORY METHOD ---
* 
* While this approach is extremely common in game development because it is 
* simple and intuitive, it has significant architectural flaws as the project grows:
* 
* 1. Violation of the Open/Closed Principle (OCP):
*    This is the biggest flaw. SOLID principles state that code should be 
*    "open for extension, but closed for modification". 
*    If we want to add a Level 4 with a new "FireMeteor", we are FORCED to 
*    open this file and modify the existing 'switch' statement inside the 
*    MeteorFactory class. We are modifying existing, working code, which 
*    increases the risk of introducing bugs.
* 
* 2. Scalability Issues (The "God Object" Anti-pattern):
*    Imagine a game with 100 levels and 50 different enemy types. The 
*    createMeteor() method would become a massive, hundreds-of-lines-long 
*    switch statement. The factory becomes a "God Object" that knows about 
*    every single meteor type in the entire game, tightly coupling all 
*    dependencies into one file.
* 
* 3. Lack of Polymorphic Delegation:
*    The Gang of Four (GoF) Factory Method pattern solves these issues by 
*    making the Factory an abstract base class, and letting subclass factories 
*    (e.g., Level1Factory, Level2Factory) decide what to instantiate. Here, 
*    creation is hardcoded into a single class's logic.
* 
* Conclusion for Students:
* Use the Simple Factory for small, limited scopes. But if your game or app 
* will expand with many new types over time, you should upgrade to the true 
* GoF Factory Method to keep your code modular, scalable, and safe.
* ============================================================================
*/
#include <iostream>
#include <memory>
#include <string>
// ==========================================
// 1. Abstract Product
// ==========================================
class Meteor {
public:
virtual ~Meteor() = default;
virtual void showInfo() const = 0;
};
// ==========================================
// 2. Concrete Products
// ==========================================
class SmallMeteor : public Meteor {
public:
void showInfo() const override {
std::cout << " -> Meteor spawned [Size: Small | Speed: Slow]" << std::endl;
}
};
class MediumMeteor : public Meteor {
public:
void showInfo() const override {
std::cout << " -> Meteor spawned [Size: Medium | Speed: Normal]" << std::endl;
}
};
class LargeMeteor : public Meteor {
public:
void showInfo() const override {
std::cout << " -> Meteor spawned [Size: Large | Speed: Fast]" << std::endl;
}
};
// ==========================================
// 3. The Simple Factory (Concrete Class)
// ==========================================
class MeteorFactory {
private:
int currentLevel;
public:
MeteorFactory() : currentLevel(1) {}
// Method to update the internal state of the factory
void setLevel(int level) {
currentLevel = level;
}
// The creation method uses hardcoded logic (a switch) to decide what to build.
// WARNING: Adding Level 4 requires modifying this exact method (Violates OCP).
std::unique_ptr<Meteor> createMeteor() const {
switch (currentLevel) {
case 1:
return std::make_unique<SmallMeteor>();
case 2:
return std::make_unique<MediumMeteor>();
case 3:
return std::make_unique<LargeMeteor>();
default:
std::cout << " [Warning] Unknown level! Defaulting to Small Meteor.\n";
return std::make_unique<SmallMeteor>();
}
}
};
// ==========================================
// 4. Game Logic / Client Code
// ==========================================
// In this version, the function takes a constant reference to the single factory.
// It doesn't take ownership (no std::unique_ptr or std::move needed for the factory), 
// because the exact same factory object will be reused for all levels.
void playLevel(const MeteorFactory& factory) {
std::cout << " [Game Engine] Spawning meteors for this level...\n";
// We use the '.' operator because 'factory' is a reference to the object.
std::unique_ptr<Meteor> m1 = factory.createMeteor();
std::unique_ptr<Meteor> m2 = factory.createMeteor();
m1->showInfo();
m2->showInfo();
std::cout << " [Game Engine] Level Cleared!\n\n";
}
// ==========================================
// 5. Main Flow
// ==========================================
int main() {
std::cout << "=== SPACE METEOR DEFENSE (SIMPLE FACTORY) ===\n\n";
// We instantiate exactly ONE factory object for the entire game session.
MeteorFactory gameFactory;
// --- LEVEL 1 ---
std::cout << "--- STARTING LEVEL 1 ---\n";
gameFactory.setLevel(1);
playLevel(gameFactory);
// Passing the factory by reference
// --- LEVEL 2 ---
std::cout << "--- STARTING LEVEL 2 ---\n";
// We just tell the SAME factory that the level has changed.
gameFactory.setLevel(2);
playLevel(gameFactory); 
// --- LEVEL 3 ---
std::cout << "--- STARTING LEVEL 3 ---\n";
gameFactory.setLevel(3);
playLevel(gameFactory);
std::cout << "=== CONGRATULATIONS! YOU WIN! ===\n";
}
//================================================================================ END
/**
* Author: Mario Galindo Queralt, Ph.D.
* Game Concept & Example Overview:
*
* This is a multi-level space game where the difficulty progressively 
* increases as the player advances. In each new level, the game generates 
* meteorites that are larger and travel at faster initial speeds, making 
* them increasingly challenging to deal with.
*
* This code example is intentionally kept simple and focuses exclusively 
* on demonstrating the "Factory Method" design pattern (as defined by the 
* Gang of Four). It illustrates how to elegantly manage the creation of 
* different meteor types as the level increases, without tightly coupling 
* the game logic to concrete meteor classes. 
*
* Instead of implementing a full game engine with graphics and collision 
* detection, this program serves as a summary of what would happen in a 
* much longer, complete game. Whenever a meteor is spawned, the program 
* simply calls its showInfo() method, which prints a text description of 
* the meteor's size and speed to the console.
*/
#include <iostream>
#include <memory>
#include <string>
// ==========================================
// 1. Abstract Product
// ==========================================
class Meteor {
public:
virtual ~Meteor() = default;
virtual void showInfo() const = 0;
};
// ==========================================
// 2. Concrete Products
// ==========================================
class SmallMeteor : public Meteor {
public:
void showInfo() const override {
std::cout << " -> Meteor spawned [Size: Small | Speed: Slow]" << std::endl;
}
};
class MediumMeteor : public Meteor {
public:
void showInfo() const override {
std::cout << " -> Meteor spawned [Size: Medium | Speed: Normal]" << std::endl;
}
};
class LargeMeteor : public Meteor {
public:
void showInfo() const override {
std::cout << " -> Meteor spawned [Size: Large | Speed: Fast]" << std::endl;
}
};
// ==========================================
// 3. Abstract Creator
// ==========================================
class FactoryMethod {
public:
virtual ~FactoryMethod() = default;
// The Factory Method
virtual std::unique_ptr<Meteor> createMeteor() const = 0;
};
// ==========================================
// 4. Concrete Creators (Levels)
// ==========================================
class FactoryLevel1 : public FactoryMethod {
public:
std::unique_ptr<Meteor> createMeteor() const override {
return std::make_unique<SmallMeteor>();
}
};
class FactoryLevel2 : public FactoryMethod {
public:
std::unique_ptr<Meteor> createMeteor() const override {
return std::make_unique<MediumMeteor>();
}
};
class FactoryLevel3 : public FactoryMethod {
public:
std::unique_ptr<Meteor> createMeteor() const override {
return std::make_unique<LargeMeteor>();
}
};
// ==========================================
// 5. Game Logic / Client Code
// ==========================================
// Passing std::unique_ptr by value transfers ownership to this function.
// The function does not need to know the specific factory or level type.
void playLevel(std::unique_ptr<FactoryMethod> factory) {
std::cout << " [Game Engine] Spawning meteors for this level...\n";
// std::unique_ptr overloads the '->' operator, allowing us to safely 
// access the methods of the managed FactoryMethod object.
std::unique_ptr<Meteor> m1 = factory->createMeteor();
std::unique_ptr<Meteor> m2 = factory->createMeteor();
m1->showInfo();
m2->showInfo();
std::cout << " [Game Engine] Level Cleared!\n\n";
}
// 'factory' goes out of scope here. The managed object is destroyed automatically.
// ==========================================
// 6. Main Flow
// ==========================================
int main() {
std::cout << "=== SPACE METEOR DEFENSE ===\n\n";
std::unique_ptr<FactoryMethod> currentFactory;
// --- LEVEL 1 ---
std::cout << "--- STARTING LEVEL 1 ---\n";
currentFactory = std::make_unique<FactoryLevel1>();
// We use std::move() to explicitly transfer ownership of the factory.
// After this call, 'currentFactory' becomes nullptr.
playLevel(std::move(currentFactory)); 
// --- LEVEL 2 ---
std::cout << "--- STARTING LEVEL 2 ---\n";
// We can safely reuse the variable by assigning a new factory object.
currentFactory = std::make_unique<FactoryLevel2>();
playLevel(std::move(currentFactory)); 
// --- LEVEL 3 ---
std::cout << "--- STARTING LEVEL 3 ---\n";
// We can also create and pass the unique_ptr directly in one line.
// The compiler implicitly handles the move semantics for temporary objects.
playLevel(std::make_unique<FactoryLevel3>());
std::cout << "=== CONGRATULATIONS! YOU WIN! ===\n";
}
//================================================================================ END
/**
* ============================================================================
*                   --- THE ABSTRACT FACTORY PATTERN ---
* Author: Mario Galindo Queralt, Ph.D.
*
* Concept:
* This program demonstrates how to create families of related objects 
* that must always work together.
* 
* Game Concept:
* This program simulates a multi-environment game. The game can take place 
* either in "Outer Space" or "Deep Underwater". Each environment requires 
* a specific "Family" of objects that are conceptually related:
*   1. A Player Vehicle (Spaceship or Submarine)
*   2. Environmental Hazards (Meteors or Sharks)
* 
* The Architecture:
* The Abstract Factory pattern is used to provide an interface for creating 
* families of related or dependent objects without specifying their concrete 
* classes. This ensures that the game engine never accidentally mixes 
* incompatible objects (like spawning a Shark in Outer Space).
* 
* Key Components of the Pattern in this Code:
*    1. Abstract Products (Player, Hazard): Interfaces for the game elements.
*    2. Concrete Products (Spaceship, Meteor, etc.): Specific implementations 
*       for each environment.
*    3. Abstract Factory (EnvironmentFactory): A base interface that declares 
*       creation methods for all abstract products.
*    4. Concrete Factories (SpaceEnvironment, UnderwaterEnvironment): These 
*       classes implement the creation logic for one specific theme/environment.
* 
* Smart Pointer & Memory Management:
* We use std::unique_ptr for automatic memory management. In this version, 
* we create two separate Hazard objects (hazard1 and hazard2) to demonstrate 
* that the factory generates independent instances. When the runEnvironment 
* function ends, the destructors will trigger, showing the automatic 
* cleanup of each individual object.
* ============================================================================
*/
#include <iostream>
#include <memory>
#include <string>
// ==========================================
// 1. Abstract Products
// ==========================================
class Player {
public:
virtual ~Player() = default;
virtual void spawn() const = 0;
};
class Hazard {
public:
virtual ~Hazard() = default; 
virtual void spawn() const = 0;
};
// ==========================================
// 2. Space Environment Implementation
// ==========================================
class Spaceship : public Player {
public:
void spawn() const override {
std::cout << " [Player] Spaceship is ready for takeoff!" << std::endl;
}
~Spaceship() {
std::cout << " [Cleanup] Spaceship removed from memory." << std::endl;
}
};
class Meteor : public Hazard {
public:
void spawn() const override {
std::cout << " [Hazard] A Meteor is drifting in the sector!" << std::endl;
}
~Meteor() {
std::cout << " [Cleanup] Meteor removed from memory." << std::endl;
}
};
// ==========================================
// 3. Underwater Environment Implementation
// ==========================================
class Submarine : public Player {
public:
void spawn() const override {
std::cout << " [Player] Submarine is diving to the ocean floor!" << std::endl;
}
~Submarine() {
std::cout << " [Cleanup] Submarine removed from memory." << std::endl;
}
};
class Shark : public Hazard {
public:
void spawn() const override {
std::cout << " [Hazard] A Shark is approaching the area!" << std::endl;
}
~Shark() {
std::cout << " [Cleanup] Shark removed from memory." << std::endl;
}
};
// ==========================================
// 4. Abstract Factory
// ==========================================
class EnvironmentFactory {
public:
virtual ~EnvironmentFactory() = default;
virtual std::unique_ptr<Player> createPlayer() const = 0;
virtual std::unique_ptr<Hazard> createHazard() const = 0;
};
// ==========================================
// 5. Concrete Factories
// ==========================================
class SpaceEnvironment : public EnvironmentFactory {
public:
std::unique_ptr<Player> createPlayer() const override {
return std::make_unique<Spaceship>();
}
std::unique_ptr<Hazard> createHazard() const override {
return std::make_unique<Meteor>();
}
~SpaceEnvironment(){
std::cout << " [Cleanup] SpaceEnvironmentFactory removed from memory." << std::endl;
}
};
class UnderwaterEnvironment : public EnvironmentFactory {
public:
std::unique_ptr<Player> createPlayer() const override {
return std::make_unique<Submarine>();
}
std::unique_ptr<Hazard> createHazard() const override {
return std::make_unique<Shark>();
}
~UnderwaterEnvironment(){
std::cout << " [Cleanup] UnderwaterEnvironmentFactory removed from memory." << std::endl;
}
};
// ==========================================
// 6. Generic Game Engine Subroutine
// ==========================================
void runEnvironment(std::unique_ptr<EnvironmentFactory> factory) {
std::cout << " --- Initializing Environment Assets ---" << std::endl;
// We create one player
std::unique_ptr<Player> player = factory->createPlayer();
// We use the factory twice to create TWO independent hazard objects
// This demonstrates that the factory is a generator of new instances.
std::unique_ptr<Hazard> hazard1 = factory->createHazard();
std::unique_ptr<Hazard> hazard2 = factory->createHazard();
// Spawn all elements
player->spawn();
// Each hazard is a separate object in memory
std::cout << " (Hazard 1):";
hazard1->spawn();
std::cout << " (Hazard 2):";
hazard2->spawn();
std::cout << " --- Gameplay simulation in progress... ---" << std::endl;
std::cout << " --- Environment Mission Completed! ---" << std::endl;
// At this point, the function ends. 'player', 'hazard1', 'hazard2', 
// and 'factory' are destroyed. You will see four separate cleanup messages!
}
// ==========================================
// 7. Main Entry Point
// ==========================================
int main() {
std::cout << "=== VIRTUAL ENVIRONMENT SIMULATOR ===\n" << std::endl;
// 1. Run Space Environment
std::cout << ">>> Setting up Space Mission..." << std::endl;
runEnvironment(std::make_unique<SpaceEnvironment>());
std::cout << "\n------------------------------------------\n" << std::endl;
// 2. Run Underwater Environment
std::cout << ">>> Setting up Underwater Mission..." << std::endl;
runEnvironment(std::make_unique<UnderwaterEnvironment>());
std::cout << "\n=== SYSTEM SHUTDOWN COMPLETE ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Prototype_traditional.cpp (Traditional Implementation)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the classic GoF Prototype pattern, specifically 
* implemented through what is known in C++ as the "Virtual Constructor" idiom.
* 
* --- THE PROBLEM:
* In C++, constructors cannot be virtual. If you hold a pointer to a base 
* class (e.g., 'Base*'), you cannot simply create a copy of the underlying 
* object (which might be a 'Derived' type) without knowing its concrete 
* class at the call site. This creates tight coupling and prevents the 
* polymorphic duplication of objects.
* 
* --- THE SOLUTION (VIRTUAL CONSTRUCTOR):
* The Prototype pattern delegates the responsibility of duplication to the 
* objects themselves. By defining a virtual 'clone()' method in the base 
* class and overriding it in every derived class, each object acts as its 
* own factory. 
* 
* --- TECHNICAL MECHANICS:
* 1. Polymorphic Copying: The 'clone()' method returns a new instance of 
*    the same type as the object it is called on, but through a base pointer.
* 2. Smart Pointers: We use 'std::unique_ptr' to ensure that the newly 
*    cloned objects are managed via RAII, preventing memory leaks.
* 3. Manual Overrides: This "Traditional" version requires every new 
*    derived class to manually implement 'clone()'.
* ============================================================================
*/
#include <iostream>
#include <memory>
struct Base
{
virtual ~Base() 
{
std::cout << " [Cleanup] Base object destroyed.\n";
}
virtual void print() const 
{
std::cout << " -> Object type: Base\n";
}
// The Virtual Constructor (Prototype)
virtual std::unique_ptr<Base> clone() const
{
return std::make_unique<Base>(*this);
// Copy current object
}
};
struct Derived : Base
{
~Derived() override 
{
std::cout << " [Cleanup] Derived object destroyed.\n";
}
// Correct syntax: 'const' goes before 'override'
void print() const override 
{
std::cout << " -> Object type: Derived\n";
}
// Overriding clone to return a new Derived object.
// Note: When using unique_ptr, the return type must match the base exactly.
std::unique_ptr<Base> clone() const override
{
return std::make_unique<Derived>(*this);
// Copy current derived object
}
};
/**
* Global helper function to demonstrate the Virtual Constructor.
* It doesn't know the concrete type of 'b', but it clones it correctly.
*/
std::unique_ptr<Base> createClone(const Base* b)
{
return b->clone();
// Calls the virtual clone() method
}
void printInfo(const Base* b)
{
b->print();
// Calls the virtual print() method
}
int main()
{
std::cout << "=== PROTOTYPE PATTERN SIMULATION ===\n\n";
{
std::cout << "Creating a Base prototype:\n";
std::unique_ptr<Base> base_1 = std::make_unique<Base>();
printInfo(base_1.get());
std::cout << "Cloning Base:\n";
auto base_2 = createClone(base_1.get()); 
printInfo(base_2.get());
std::cout << "--- Base objects going out of scope ---\n";
}
// Destructors are called automatically here
std::cout << "\n------------------------------------------\n\n";
{
std::cout << "Creating a Derived prototype:\n";
std::unique_ptr<Base> derived_1 = std::make_unique<Derived>();
printInfo(derived_1.get());
std::cout << "Cloning Derived (via Base pointer in createClone):\n";
// This is the magic: createClone receives Base* but clones a Derived object.
auto derived_2 = createClone(derived_1.get()); 
printInfo(derived_2.get());
std::cout << "--- Derived objects going out of scope ---\n";
}
// Destructors are called automatically here
std::cout << "\n=== END OF SIMULATION ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Prototype_deducing.cpp (Modern C++23 Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program implements a cutting-edge version of the Prototype pattern 
* using C++23 "Deducing This" (Explicit Object Parameters). It represents 
* the pinnacle of the pattern's evolution in C++.
* 
* --- THE ARCHITECTURAL EVOLUTION (INTERFACE PATTERN):
* 1. Infrastructure Layer (Mixin): 'SmartCloneable' is a non-template utility 
*    that provides the static cloning machinery.
* 
* 2. Interface Layer (Base): 'Base' is now a pure Interface (ABC). It defines 
*    the contract for the domain but cannot be instantiated. Crucially, 
*    cloning a 'Base' directly is impossible because abstract types cannot 
*    be created by the Mixin's 'make_unique' call.
* 
* 3. Implementation Layer (Derived): Concrete classes implement the interface 
*    and leverage the Mixin to satisfy the virtual cloning contract with 
*    zero boilerplate.
*
* --- TECHNICAL MECHANICS:
* - Deducing This: 'clone(this auto self)' deduces the derived type at the 
*   call site.
* - Static Safety: Attempting to clone an abstract 'Base' results in a 
*   compile-time error, as 'decltype(self)' would be an abstract type.
* - Polymorphism: 'clone_polymorphic()' bridges the static machinery with 
*   traditional dynamic dispatch.
* ============================================================================
*/
#include <iostream>
#include <memory>
#include <string>
//--------------------------------------------------------- Prototype Mixin:
class SmartCloneable {
public:
virtual ~SmartCloneable() = default;
/**
* The 'this auto self' parameter captures the concrete type.
* If called on an abstract type, std::make_unique will fail to compile,
* providing perfect architectural enforcement.
* Guideline: "this auto self" (pass by value) ensures a deep copy is 
* automatically created as a function parameter.
*/
auto clone(this auto self) {
return std::make_unique<decltype(self)>(std::move(self));
}
};
//--------------------------------------------------------- Base Interface:
class Base : public SmartCloneable {
public:
// Pure Interface: No data, only contract.
virtual ~Base() {
std::cout << " [Cleanup] Interface Base destroyed.\n";
}
virtual void print() const = 0;
// Pure Virtual
/**
* The Virtual Constructor Contract:
* Must be implemented by concrete classes.
*/
virtual std::unique_ptr<Base> clone_polymorphic() const = 0;
};
//------------------------------------------------------- Concrete Derived:
class Derived : public Base {
public:
Derived() = default;
Derived(const Derived& other) : Base(other) {}
~Derived() override {
std::cout << " [Cleanup] Concrete Derived destroyed.\n";
}
void print() const override {
std::cout << " -> Object type: Concrete Derived\n";
}
/**
* Implementation using the Mixin's static logic.
* Returns a unique_ptr<Derived> which is then upcasted to unique_ptr<Base>.
*/
std::unique_ptr<Base> clone_polymorphic() const override {
return this->clone();
}
};
//------------------------------------------------------------------- Main:
int main() {
std::cout << "=== PROTOTYPE PATTERN SIMULATION: C++23 DEDUCING THIS ===\n";
{
std::cout << "\n--- PHASE 1: Polymorphic Cloning via Interface ---\n";
// We cannot do: Base b; or b.clone(); -> It would not compile.
std::unique_ptr<Base> original = std::make_unique<Derived>();
std::cout << "Original object (managed via Interface pointer):\n";
original->print();
std::cout << "Cloning via virtual contract:\n";
auto copy = original->clone_polymorphic();
copy->print();
std::cout << "--- Objects going out of scope ---\n";
} 
std::cout << "\n----------------------------------------------------------\n";
{
std::cout << "\n--- PHASE 2: Static Covariance (Direct Access) ---\n";
Derived original;
original.print();
std::cout << "Cloning via Deducing This:\n";
// Returns std::unique_ptr<Derived> directly.
auto specificClone = original.clone();
specificClone->print();
std::cout << "--- Objects going out of scope ---\n";
} 
std::cout << "\n=== END OF SIMULATION ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Singleton.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- EDUCATIONAL NOTE:
* "We use the Heap (via unique_ptr) when the Singleton is very large, when we
* want to use polymorphism (deciding the implementation at runtime) or when we
* need total control over dynamic memory. We use Meyers' Singleton for
* lightweight and simple Singletons due to its elegance and native thread
* safety."
*
* --- CONFIGURATION:
* Toggle the definition below to switch between implementations.
* Only ONE implementation of getInstance() will be compiled at a time,
* preserving the "Single Instance" rule of the pattern.
* ============================================================================
*/
#include <iostream>
#include <memory>
// Comment or uncomment this line to switch between Meyers' and Heap version
//#define USE_HEAP_SINGLETON
class Singleton
{
private:
// Private constructor prevents direct instantiation.
Singleton()
{
std::cout << " [System] Singleton instance created at address: " << this << "\n";
}
public:
// Rule of Three/Five: Delete copy constructor and assignment operator.
Singleton(const Singleton&) = delete;
Singleton& operator=(const Singleton&) = delete;
virtual ~Singleton()
{
std::cout << " [System] Singleton instance destroyed.\n";
}
/**
* The Singleton Access Point
* The logic changes based on the #define at the top of the file.
*/
static Singleton& getInstance()
{
#ifdef USE_HEAP_SINGLETON
// APPROACH: Heap-based Singleton (Dynamic Memory)
// Managed by std::unique_ptr for automatic cleanup.
// We use 'new' because make_unique cannot access a private constructor.
static std::unique_ptr<Singleton> instance(new Singleton());
std::cout << " [Info] Using Heap-based implementation (unique_ptr).\n";
return *instance;
#else
// APPROACH: Meyers' Singleton (Static/Data Segment)
// Simplest, thread-safe (C++11+), and stack-efficient.
static Singleton instance;
std::cout << " [Info] Using Meyers' implementation (Static Segment).\n";
return instance;
#endif
}
void talk() const
{
std::cout << " -> Hello! I am the only instance of this class.\n";
}
};
// ==========================================
// Main Execution Flow
// ==========================================
int main()
{
std::cout << "=== SINGLETON IMPLEMENTATION TEST ===\n\n";
#ifdef USE_HEAP_SINGLETON
std::cout << "CONFIGURATION: HEAP MODE ENABLED\n";
#else
std::cout << "CONFIGURATION: MEYERS MODE ENABLED\n";
#endif
std::cout << "-------------------------------------\n\n";
// First access: The object is created here.
std::cout << "Requesting instance for the first time...\n";
Singleton& s1 = Singleton::getInstance();
s1.talk();
// Second access: Returns the existing instance.
std::cout << "\nRequesting instance for the second time...\n";
Singleton& s2 = Singleton::getInstance();
s2.talk();
// Verification: Both pointers must be identical.
std::cout << "\nVerification:\n";
std::cout << " Address of s1: " << &s1 << "\n";
std::cout << " Address of s2: " << &s2 << "\n";
if (&s1 == &s2)
{
std::cout << " SUCCESS: Identity confirmed. Only one instance exists.\n";
}
else
{
std::cout << " FAILURE: Multiple instances detected!\n";
}
std::cout << "\n=== END OF MAIN ===\n";
// Cleanup happens after main() returns as the program terminates.
}
//================================================================================ END
/*
* ============================================================================
* File: Singleton.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This implementation demonstrates the "Multiton" pattern combined with a
* Registry mechanism. Unlike a standard Singleton (which manages one instance),
* this pattern manages a collection of named instances.
*
* --- ENCAPSULATED AUTO-REGISTRATION:
* Registration is an encapsulated concern of each concrete class. We use a
* private static boolean member initialized by an immediate lambda to trigger
* the registration BEFORE the 'main()' function starts.
*
* --- COMPILATION & ACCESS CONTROL:
* By defining the static member's initialization after the class is fully
* defined, we avoid "incomplete type" errors. This approach allows the
* constructors to remain 'private', ensuring that instances are only created
* through the controlled, automatic registration process.
*
* --- ARCHITECTURAL NOTE:
* This approach achieves decoupling: the registry handles the lifecycle
* and lookup, while the client simply requests instances by a unique
* identifier. This is a powerful technique for modular systems where
* components can register themselves dynamically.
*
* --- MEMORY MANAGEMENT & STORAGE:
* The Singleton instances in this example are stored in the **Static Data
* Segment** (specifically the .data or .bss sections of the executable).
* Unlike stack variables, they are not destroyed when a function ends;
* and unlike heap variables, they do not require 'new' or 'delete'. They are
* allocated by the system loader when the program starts and persist until
* the process terminates.
* ============================================================================
*/
#include <string>
#include <string_view>
#include <map>
#include <iostream>
#include <stdexcept>
//--------------------------------------------------------------------- Singleton Base
class Singleton
{
private:
static inline std::map<std::string, Singleton*> singletonMap;
protected:
static Singleton *getSingleton(std::string_view name)
{
auto it = singletonMap.find(std::string(name));
if(it == singletonMap.end()) return nullptr;
return it->second;
}
public:
virtual ~Singleton() = default;
Singleton(const std::string& name)
{
auto [it, success] = singletonMap.insert({name, this});
if(!success) throw std::invalid_argument("Duplicate singleton name: " + name);
}
Singleton(const Singleton&) = delete;
Singleton& operator=(const Singleton&) = delete;
};
//---------------------------------------------------------------------------- Service
class Service : public Singleton
{
public:
using Singleton::Singleton;
virtual ~Service() = default;
static Service *getService(std::string_view name)
{
return static_cast<Service*>(getSingleton(name));
}
virtual void method_1() { std::cout << "Default method_1\n"; }
virtual void method_2() { std::cout << "Default method_2\n"; }
};
//------------------------------------------------------------------------ Singleton A
class Singleton_A final : public Service
{
private:
Singleton_A(const std::string& name) : Service{name}
{
std::cout << " [System] Singleton_A registered as a Service.\n";
}
~Singleton_A() override { std::cout << " [System] Singleton_A destroyed.\n"; }
void method_1() override { std::cout << " -> Running Singleton_A::method_1\n"; }
static bool registered_;
};
bool Singleton_A::registered_ = []()
{
static Singleton_A instance{"Singleton_A"};
return true;
}();
//------------------------------------------------------------------------ Singleton B
class Singleton_B final : public Service
{
private:
Singleton_B(const std::string& name) : Service{name}
{
std::cout << " [System] Singleton_B registered as a Service.\n";
}
~Singleton_B() override { std::cout << " [System] Singleton_B destroyed.\n"; }
void method_2() override { std::cout << " -> Running Singleton_B::method_2\n"; }
static bool registered_;
};
bool Singleton_B::registered_ = []()
{
static Singleton_B instance{"Singleton_B"};
return true;
}();
//------------------------------------------------------------------------------- Main
int main()
{
std::cout << "=== SINGLETON WITH ENCAPSULATED AUTO-REGISTER ===\n" << std::endl;
Service *sA1 = Service::getService("Singleton_A");
if(sA1) sA1->method_1();
Service *sB = Service::getService("Singleton_B");
if(sB) sB->method_2();
Service *sC = Service::getService("Singleton_C");
if(!sC) std::cout << " [Error] Singleton_C not found in registry.\n";
std::cout << "\nRequesting Singleton A instance for the second time...\n";
Service *sA2 = Service::getService("Singleton_A");
std::cout << "\nVerification:\n";
std::cout << " Address of sA1: " << sA1 << "\n";
std::cout << " Address of sA2: " << sA2 << "\n";
if(sA1 == sA2)
{
std::cout << " SUCCESS: Identity confirmed. Only one instance exists.\n";
}
else
{
std::cout << " FAILURE: Multiple instances detected!\n";
}
std::cout << "\n=== END OF MAIN ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Singleton_Deducing.cpp (Modern C++23 Version)
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program demonstrates a Singleton Hierarchy using C++23 "Deducing This"
* (Explicit Object Parameters).
*
* --- THE ARCHITECTURAL PROBLEM:
* When Singletons inherit from a common Base class (e.g., a generic Service
* Manager), calling a Base method traditionally returns a reference to the
* Base class. This breaks the "Fluent Interface" because the compiler
* "forgets" the derived type, preventing further calls to derived-specific
* methods in the same chain.
*
* --- THE C++23 SOLUTION:
* By using 'this auto&& self' in the Base class methods, we can capture the
* exact type of the calling Singleton at compile-time. This allows the Base
* class to return a reference to the Derived type automatically, maintaining
* the chain without using CRTP or virtual functions.
*
* --- TECHNICAL MECHANICS:
* 1. Base Utility: 'ManagerBase' provides common logic (e.g., 'initialize')
*    to all services.
* 2. Static Dispatch: 'self' in 'initialize(this auto&& self)' deduces
*    whether it's being called by 'Logger' or 'Database'.
* 3. Meyer's Singleton: We still use the safe, thread-safe static local
*    instance within each derived class.
* ============================================================================
*/
#include <iostream>
#include <string>
//------------------------------------------------- Common Service Manager:
/**
* The Base class is NO LONGER a template.
* "Deducing This" handles the polymorphic return types.
*/
class ManagerBase {
public:
// Common initialization logic for all Singletons
auto&& initialize(this auto&& self) {
std::cout << " [Base] Global Service Initialization sequence started...\n";
// Imagine complex setup logic here...
return self;
// Returns the derived Singleton!
}
auto&& setLogLevel(this auto&& self, int level) {
std::cout << " [Base] Setting system-wide log level to: " << level << "\n";
return self;
}
};
//------------------------------------------------------- Logger Singleton:
class Logger : public ManagerBase {
public:
// Standard Meyer's Singleton access
static Logger& instance() {
static Logger inst;
return inst;
}
// --- THE RULE OF SEVEN (Singleton constraints) ---
Logger(const Logger&)            = delete;
Logger(Logger&&)                 = delete;
Logger& operator=(const Logger&) = delete;
Logger& operator=(Logger&&)      = delete;
// Specific Logger behavior
Logger& log(const std::string& msg) {
std::cout << " [Logger] Event: " << msg << "\n";
return *this;
}
private:
Logger() { std::cout << " [Constructor] Logger Service Ready.\n"; }
};
//----------------------------------------------------- Database Singleton:
class Database : public ManagerBase {
public:
static Database& instance() {
static Database inst;
return inst;
}
// --- THE RULE OF SEVEN ---
Database(const Database&)            = delete;
Database(Database&&)                 = delete;
Database& operator=(const Database&) = delete;
Database& operator=(Database&&)      = delete;
// Specific Database behavior
Database& connect(const std::string& connectionString) {
std::cout << " [Database] Connected to " << connectionString << "\n";
return *this;
}
private:
Database() { std::cout << " [Constructor] Database Service Ready.\n"; }
};
//------------------------------------------------------------------- Main:
int main() {
std::cout << "=== SINGLETON HIERARCHY: C++23 DEDUCING THIS ===\n\n";
/**
* MAGIC AT WORK:
* 'Logger::instance().initialize()' returns a 'Logger&' even though
* 'initialize' is defined in 'ManagerBase'.
* This allows us to chain '.log()' immediately.
*/
std::cout << "--- Testing Logger Chain ---\n";
Logger::instance()
.initialize()
.setLogLevel(2)
.log("Singleton hierarchy is working perfectly.");
std::cout << "\n--- Testing Database Chain ---\n";
/**
* Here, 'initialize' returns a 'Database&', allowing the call to '.connect()'.
*/
Database::instance()
.initialize()
.connect("prod_server_db_01");
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Lazy_Traditional.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Lazy Initialization pattern. The creation of 
* heavy 'Fruit' objects is deferred until the first time they are requested.
* 
* --- LAZY INITIALIZATION:
* We use an 'unordered_map' to cache instances. If a requested fruit does not 
* exist, it is created (Lazy) and stored (Cached).
* 
* --- MEMORY MANAGEMENT:
* We use 'std::shared_ptr' because 'Fruit' instances are shared resources. 
* The factory acts as the primary owner, but clients can hold their own 
* shared references to these objects safely.
* ============================================================================
*/
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <memory>
class Fruit
{
private:
std::string type_;
// Note: Constructor is private to force the use of the static factory method
explicit Fruit(std::string type) : type_{std::move(type)}
{
std::cout << " [System] Fruit instance created: " << type_ << "\n";
}
public:
~Fruit()
{
std::cout << " [Cleanup] Fruit instance destroyed: " << type_ << "\n";
}
// Lazy Factory method: gets the Fruit instance associated with a certain type.
// Creates new ones only as needed (Lazy Initialization).
static std::shared_ptr<Fruit> getFruit(std::string_view type);
static void printCurrentTypes()
{
std::cout << " Number of shared instances: " << cache_.size() << "\n";
for(const auto& [type, fruit] : cache_) std::cout << " - " << type << "\n";
std::cout << "\n";
}
private:
// Using shared_ptr as the cache owner allows shared access by clients
static inline std::map<std::string, std::shared_ptr<Fruit>> cache_;
};
// Definition of the factory method
std::shared_ptr<Fruit> Fruit::getFruit(std::string_view type)
{
std::string key{type};
auto it = cache_.find(key);
if (it == cache_.end())
{
auto fruit = std::shared_ptr<Fruit>(new Fruit(key));
cache_[key] = fruit;
return fruit;
}
return it->second;
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== LAZY INITIALIZATION PATTERN ===\n" << std::endl;
// Verify a clean cache:
std::cout << "Requesting number of shared instances in the cache:\n";
Fruit::printCurrentTypes();
// 1st request: Creates a new "Banana"
std::cout << "Requesting 'Banana' (1st time):\n";
auto f1 = Fruit::getFruit("Banana");
Fruit::printCurrentTypes();
// 2nd request: Creates a new "Apple"
std::cout << "Requesting 'Apple' (1st time):\n";
auto f2 = Fruit::getFruit("Apple");
Fruit::printCurrentTypes();
// 3rd request: Returns pre-existing "Banana" instance from the cache
std::cout << "Requesting 'Banana' (2nd time):\n";
auto f3 = Fruit::getFruit("Banana");
Fruit::printCurrentTypes();
std::cout << "=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Lazy_Deducing_this.cpp (Modern C++23 Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates a "Transparent Object-level" Lazy Initialization 
* using the C++23 "Deducing This" feature. It allows heavy objects to defer 
* resource-intensive loading until the exact moment a business method is 
* invoked, completely hiding this complexity from the client.
* 
* --- THE ARCHITECTURAL EVOLUTION:
* 1. Factory-Level (Example 01): Focuses on the "when" of object creation 
*    using a global cache and static factories.
* 2. Object-Level (Example 02): Focuses on the "internal state" of the object. 
*    The object exists as a lightweight shell, only "heavying up" its internal 
*    resources upon the first functional call.
* 
* --- THE C++23 REVOLUTION (ZERO-OVERHEAD POLYMORPHISM):
* Traditionally, a Base class could only trigger logic in a Derived class 
* using either:
*   A) Virtual Functions: Incurs the cost of a VTable and dynamic dispatch.
*   B) CRTP: Incurs complex, circular template syntax and increased code bloat.
* 
* C++23 "Deducing This" enables "Static Dispatch" with a clean, non-template 
* hierarchy. The 'LazyComponent' base class remains a simple class, yet its 
* methods can "reach down" into the derived class at compile-time. This 
* achieves the same goal as virtual functions but with zero runtime overhead 
* and much simpler code.
* 
* --- TECHNICAL MECHANICS:
* 1. Mixin Infrastructure: 'LazyComponent' provides 'ensure_initialized(this 
*    auto&& self)'. 
* 2. Static Injection: The 'self' parameter deduces the 'HeavyResource' type 
*    at the call site, allowing the base to invoke private child methods 
*    (via friendship) without knowing the child's type during its own 
*    definition.
* 3. Total Transparency: The client interacts with the public API normally. 
*    The "Lazy" check is an invisible safety layer injected by the Mixin.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <utility>
//----------------------------------------------------- Infrastructure Layer:
/**
* LazyComponent Mixin:
* Provides the machinery to ensure an object is loaded exactly once.
*/
class LazyComponent {
private:
bool initialized_{false};
protected:
/**
* Using Deducing This to access the derived class's private members.
* This method is called internally by the derived class's public API.
*/
void ensure_initialized(this auto&& self) {
if(!self.initialized_) {
std::cout << " [Mixin] Lazy check: Resource not loaded. Initializing now...\n";
self.load_resources();
// Static dispatch to Derived::load_resources
self.initialized_ = true;
}
}
};
//---------------------------------------------------------- Business Layer:
class HeavyResource : public LazyComponent {
// Allows the Mixin to call private load_resources()
friend class LazyComponent;
private:
std::string name_;
int data_value_{0};
/**
* Private loading logic. 
* Only accessible via the LazyComponent infrastructure.
*/
void load_resources() {
std::cout << " [System] Loading heavy data for: " << name_ << "...\n";
data_value_ = 42;
// Emulated heavy data
}
public:
explicit HeavyResource(std::string name) : name_{std::move(name)} {
std::cout << " [System] HeavyResource created (dormant): " << name_ <<
" (Resource NOT loaded yet)\n";
}
/**
* Public API: Transparently handles initialization.
*/
void process() {
ensure_initialized();
// Internal check
std::cout << " [System] Processing data: " << data_value_ << " in " << name_ << "\n";
}
void update(int new_val) {
ensure_initialized();
// Internal check
data_value_ = new_val;
std::cout << " [System] Data updated to: " << data_value_ << " in " << name_ << "\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main() {
std::cout << "=== MODERN LAZY INITIALIZATION: DEDUCING THIS (TRANSPARENT) ===\n" << std::endl;
std::cout << "--- PHASE 1: Object Creation ---\n";
HeavyResource resource("MainDatabase");
// The object exists but its heavy resources are not yet loaded.
std::cout << "\n--- PHASE 2: Transparent Access (First Call) ---\n";
// The user doesn't know about 'ensure_initialized'. It just works.
resource.process();
std::cout << "\n--- PHASE 3: Subsequent Access (No reload) ---\n";
resource.update(100);
resource.process();
std::cout << "\n--- PHASE 4: Temporary Object Access ---\n";
// Works perfectly with rvalues too!
HeavyResource("TemporaryBuffer").process();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: SameParameters.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* It is not possible to create two constructors with the exact same 
* parameter types (e.g., float, float). To solve this ambiguity, we use 
* Named Constructors (static methods) that clearly describe the input.
* ============================================================================
*/
#include <cmath>
#include <iostream>
class Point
{
private:
float x_;
float y_;
// The true constructor is private
// It internally stores rectangular coordinates
Point(float x, float y) : x_{x}, y_{y} { } 
public:
// Named constructors:
static Point rectangular(float x, float y)
{
std::cout << " [System] A rectangular coordinate has been created.\n";
return Point(x, y);
}
static Point polar(float radius, float angle)
{
std::cout << " [System] A polar coordinate has been created.\n";
return Point(radius * std::cos(angle), radius * std::sin(angle));
}
void print() const
{
std::cout << "  -> Point(x: " << x_ << ", y: " << y_ << ")\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== NAMED CONSTRUCTORS (SAME PARAMETERS) ===\n" << std::endl;
Point p1 = Point::rectangular(5.7f, 1.2f);
p1.print();
std::cout << "\n";
Point p2 = Point::polar(5.7f, 1.2f);
p2.print();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: CreatedOnlyWithNew.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* By making all constructors private and exposing static creation methods, 
* we force objects to always be created on the heap (via 'new' wrapped in 
* a std::unique_ptr) and never on the stack.
* ============================================================================
*/
#include <iostream>
#include <memory>
class Fred
{
private:
int i_;
// The constructors themselves are private:
Fred() : i_{0} { }
explicit Fred(int i) : i_{i} { }
Fred(const Fred& other) : i_{other.i_} { }
public:
~Fred()
{
std::cout << " [Cleanup] Fred " << i_ << " destroyed (Memory freed automatically).\n";
}
// Named constructors returning safe smart pointers:
// Note: std::make_unique cannot be used here because constructors are private.
static std::unique_ptr<Fred> create()
{
return std::unique_ptr<Fred>(new Fred());
}
static std::unique_ptr<Fred> create(int i)
{
return std::unique_ptr<Fred>(new Fred(i));
}
static std::unique_ptr<Fred> create(const Fred& other)
{
return std::unique_ptr<Fred>(new Fred(other));
}
static std::unique_ptr<Fred> create(const std::unique_ptr<Fred>& other)
{
return std::unique_ptr<Fred>(new Fred(*other));
}
void talk() const
{
std::cout << "  -> Fred talking: i = " << i_ << '\n';
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== NAMED CONSTRUCTORS (FORCE HEAP ALLOCATION) ===\n" << std::endl;
// If we try: Fred p; 
// The compiler will throw an error (constructor is private).
std::cout << "Creating Freds dynamically...\n";
std::unique_ptr<Fred> f1 = Fred::create(1);
f1->talk();
auto f2 = Fred::create(2);
f2->talk();
std::cout << "\nCreating copies of Fred...\n";
std::unique_ptr<Fred> f1Copy = Fred::create(*f1);
f1Copy->talk();
auto f2Copy = Fred::create(f2);
f2Copy->talk();
std::cout << "\n=== END OF MAIN (Smart pointers will trigger cleanup) ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Adapter.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Adapter Pattern. It allows a modern 
* interface that expects coordinates (x1, y1, x2, y2) to work with a legacy 
* class that expects (x, y, width, height).
* 
* The RectangleAdapter "translates" the point-based coordinates into 
* dimension-based values required by the Legacy library.
* ============================================================================
*/
#include <iostream>
#include <memory>
//--------------------------------------------------------- Legacy (Adaptee):
// This is the old library we cannot modify.
class LegacyRectangle
{
public:
// Added 'const' to allow calls from a const Adapter.
void oldDraw(int x, int y, int w, int h) const
{
std::cout << " [Legacy Library] Drawing rectangle..." << std::endl;
std::cout << "  -> Origin: (" << x << ", " << y << ")" << std::endl;
std::cout << "  -> Dimensions: " << w << "x" << h << std::endl;
}
};
//------------------------------------------------- Modern Interface (Target):
// This is the interface our modern system understands.
class ModernRectangle
{
public:
virtual ~ModernRectangle() = default;
// Marked as 'const' to be compatible with const references.
virtual void draw(int x1, int y1, int x2, int y2) const = 0;
};
//--------------------------------------------------------- The Adapter:
// The Adapter translates Modern calls into Legacy calls.
class RectangleAdapter : public ModernRectangle
{
private:
// This is created automatically when RectangleAdapter is instantiated.
LegacyRectangle legacyInstance_;
public:
// Implementation of the modern interface.
void draw(int x1, int y1, int x2, int y2) const override
{
std::cout << " [Adapter] Converting coordinates to dimensions..." << std::endl;
int width  = x2 - x1;
int height = y2 - y1;
// Delegate the call to the internal legacy instance.
legacyInstance_.oldDraw(x1, y1, width, height);
}
};
//--------------------------------------------------------- Main Engine:
// This represents the client that only knows about the modern interface.
void render(const ModernRectangle& rect)
{
// Now this works because 'draw' is a const method.
rect.draw(10, 10, 50, 30);
}
int main()
{
std::cout << "=== ADAPTER PATTERN SIMULATION ===\n" << std::endl;
// 1. We instantiate the Adapter.
// 2. This automatically instantiates the private 'legacyInstance_' inside it.
std::unique_ptr<ModernRectangle> myRectangle = std::make_unique<RectangleAdapter>();
// The system calls 'render' which expects a ModernRectangle reference.
render(*myRectangle);
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Bridge.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Bridge Pattern by decoupling a "Shape" 
* (the Abstraction) from its "Renderer" (the Implementation).
* 
* Instead of creating VectorCircle, RasterCircle, VectorSquare, etc., 
* we create two independent hierarchies and connect them via a pointer 
* (the Bridge) inside the Shape class.
* ============================================================================
*/
#include <iostream>
#include <memory>
#include <string>
//------------------------------------------------- Implementer (Interface):
// This defines how a shape is actually drawn on a specific platform.
class Renderer
{
public:
virtual ~Renderer() = default;
virtual void renderCircle(float radius) const = 0;
virtual void renderSquare(float side) const = 0;
};
//----------------------------------------------- Concrete Implementer A:
class VectorRenderer : public Renderer
{
public:
void renderCircle(float radius) const override
{
std::cout << " [Vector] Drawing a circle of radius " << radius << std::endl;
}
void renderSquare(float side) const override
{
std::cout << " [Vector] Drawing a square of side " << side << std::endl;
}
};
//----------------------------------------------- Concrete Implementer B:
class RasterRenderer : public Renderer
{
public:
void renderCircle(float radius) const override
{
std::cout << " [Raster] Drawing pixels for a circle of radius " << radius << std::endl;
}
void renderSquare(float side) const override
{
std::cout << " [Raster] Drawing pixels for a square of side " << side << std::endl;
}
};
//------------------------------------------------- Abstraction (Base):
// This defines the high-level logic. It holds the "Bridge" to a Renderer.
class Shape
{
protected:
// The Bridge: a pointer to the implementation
std::unique_ptr<Renderer> rendererBridge_;
public:
explicit Shape(std::unique_ptr<Renderer> renderer) 
: rendererBridge_{std::move(renderer)} { }
virtual ~Shape() = default;
virtual void draw() const = 0;
virtual void resize(float factor) = 0;
};
//------------------------------------------------- Refined Abstraction 1:
class Circle : public Shape
{
private:
float radius_;
public:
Circle(std::unique_ptr<Renderer> renderer, float radius)
: Shape{std::move(renderer)}, radius_{radius} { }
void draw() const override
{
rendererBridge_->renderCircle(radius_);
}
void resize(float factor) override
{
radius_ *= factor;
}
};
//------------------------------------------------- Refined Abstraction 2:
class Square : public Shape
{
private:
float side_;
public:
Square(std::unique_ptr<Renderer> renderer, float side)
: Shape{std::move(renderer)}, side_{side} { }
void draw() const override
{
rendererBridge_->renderSquare(side_);
}
void resize(float factor) override
{
side_ *= factor;
}
};
//--------------------------------------------------------- Main Engine:
int main()
{
std::cout << "=== BRIDGE PATTERN SIMULATION ===\n" << std::endl;
// 1. Create a Vector Circle
std::unique_ptr<Shape> circle = std::make_unique<Circle>(
std::make_unique<VectorRenderer>(), 5.0f
);
// 2. Create a Raster Square
std::unique_ptr<Shape> square = std::make_unique<Square>(
std::make_unique<RasterRenderer>(), 10.0f
);
circle->draw();
square->draw();
std::cout << "\n --- Resizing Shapes ---" << std::endl;
circle->resize(2.0f);
square->resize(0.5f);
circle->draw();
square->draw();
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Employee.cpp (Pimpl Idiom / Bridge)
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program implements the Pimpl (Pointer to Implementation) idiom, which
* is a specific C++ application of the Bridge pattern.
*
* --- MEMORY MANAGEMENT:
* We use 'std::unique_ptr' to manage the 'pimpl' object. Unlike raw pointers,
* this ensures that the implementation is cleaned up automatically,
* respecting RAII principles and preventing memory leaks.
*
* --- INVARIANT:
* The 'pimpl' pointer must never be nullptr. To enforce this, we manually
* implement the move constructor and move assignment operator to ensure
* that moved-from objects are immediately re-initialized with a valid
* implementation, preventing crashes when accessed after a move operation.
*
* --- COMPILATION FIREWALL:
* The definition of 'class Employee::Impl' is hidden in this .cpp file.
* Changes to 'Impl' do not require recompilation of any client code that
* only includes 'Employee.h'.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <utility>
//---------------------------------------------------------------------------- Employe
class Employee
{
public:
Employee();
// 1 DC: Default constructor
Employee(std::string name, std::string id);
// 7 PC: Parametric Constructor
// In Pimpl with unique_ptr, we must manually define Copy logic
// because unique_ptr is move-only.
Employee(const Employee& other);
// 2 CC: Copy constructor (Deep Copy)
Employee& operator=(const Employee& other);
// 4 CA: Copy assignment
// Move logic can be defaulted, but must be defined in the .cpp
Employee(Employee&& other) noexcept;
// 3 MC: Move constructor
Employee& operator=(Employee&& other) noexcept;
// 5 MA: Move assignment
~Employee();
// Defined in .cpp where Impl is complete // 6 De: Destructor
std::string getName() const;
void setName(std::string&& name);
std::string getId() const;
void setId(std::string&& id);
private:
class Impl;
std::unique_ptr<Impl> pimpl;
// The Bridge. This pointer can't be nullptr!
};
//------------------------------------------------- Nested Struct Definition:
class Employee::Impl
{
private:
std::string name_{"No name"};
std::string id_{"No id"};
public:
Impl() = default;
// 1 DC: Default constructor
// 7 PC: Parametric Constructor
Impl(std::string name, std::string id) : name_{std::move(name)}, id_{std::move(id)} {}
// --- RULE OF FIVE vs. RULE OF ZERO ---
// According to the "Rule of Zero", we could leave all five of the following
// special member functions undefined, and the compiler would generate them
// automatically. However, per the "Rule of Five", if even one is explicitly
// defined, it is best practice to define all five to ensure consistent and
// safe behavior. In this case, we explicitly default all five functions to
// prevent compiler warnings when at least one is defined and to clarify intent,
// although defining none of them would also have been a valid approach.
// (The numbering follows my "Rule of 7" mnemonic system)
Impl(const Impl&)                = default;
// 2 CC: Copy Constructor
Impl(Impl&&) noexcept            = default;
// 3 MC: Move Constructor
Impl& operator=(const Impl&)     = default;
// 4 CA: Copy Assignment
Impl& operator=(Impl&&) noexcept = default;
// 5 MA: Move Assignment
~Impl()                          = default;
// 6 De: Destructor
std::string getName() const {return name_;}
void setName(std::string&& name) {name_ = std::move(name);}
std::string getId() const {return id_;}
void setId(std::string&& id) {id_ = std::move(id);}
};
//------------------------------------------------- Employee Implementation:
// 1 DC: Default constructor
Employee::Employee() : pimpl{std::make_unique<Impl>()}
{
std::cout << " 1 DC: Default constructor\n";
}
// 7 PC: Parametric Constructor
Employee::Employee(std::string name, std::string id)
: pimpl{std::make_unique<Impl>(std::move(name), std::move(id))}
{
std::cout << " 7 PC: Parametric Constructor\n";
}
// 2 CC: Copy constructor (Deep Copy)
Employee::Employee(const Employee& other)
: pimpl{std::make_unique<Impl>(*other.pimpl)}
{
std::cout << " 2 CC: Copy constructor\n";
}
// 4 CA: Copy assignment
Employee& Employee::operator=(const Employee& other)
{
std::cout << " 4 CA: Copy assignment\n";
if (this != &other)
{
*pimpl = *other.pimpl;
// Uses default Impl copy assignment operator
}
return *this;
}
// 3 MC: Move constructor
Employee::Employee(Employee&& other) noexcept
: pimpl{std::move(other.pimpl)}
// Transfers ownership via unique_ptr move constructor
{
std::cout << " 3 MC: Move constructor\n";
// Instead of leaving 'other.pimpl' as nullptr (default move constructor behavior),
// we re-initialize it to maintain our "Can't be nullptr" invariant.
other.pimpl = std::make_unique<Impl>();
}
// 5 MA: Move assignment
Employee& Employee::operator=(Employee&& other) noexcept
{
std::cout << " 5 MA: Move assignment\n";
if (this != &other)
{
// Using swap is an elegant way to maintain the non-null invariant
std::swap(pimpl, other.pimpl);
}
return *this;
}
// 6 De: Destructor
Employee::~Employee() = default;
std::string Employee::getName() const
{
return pimpl->getName();
// Here the bridge is used
}
void Employee::setName(std::string&& name)
{
pimpl->setName(std::move(name));
// Here the bridge is used
}
std::string Employee::getId() const
{
return pimpl->getId();
// Here the bridge is used
}
void Employee::setId(std::string&& id)
{
pimpl->setId(std::move(id));
// Here the bridge is used
}
//------------------------------------------------------------------------------- Main
int main()
{
std::cout << "=== BRIDGE (PIMPL) PATTERN SIMULATION ===\n\n";
std::cout << " 1 DC: Employee():\n";
Employee e1;
// 1 DC: Employee()
std::cout << "\n 7 PC: Employee(std::string name, std::string id):\n";
Employee e2{"Jimmy", "1-653-9"};
// 7 PC: Employee(std::string name, std::string id)
std::cout << "\n 2 CC: Employee(Employee const& other):\n";
Employee e3{e2};
// 2 CC: Employee(Employee const& other)
std::cout << "\n 3 MC: Employee(Employee&& other):\n";
Employee e4{std::move(e2)};
// 3 MC: Employee(Employee&& other)
std::cout << "\n 4 CA: operator=(Employee const& other):\n";
e2 = e4;
// 4 CA: operator=(Employee const& other)
std::cout << "\n 5 MA: operator=(Employee&& other):\n";
e1 = std::move(e4);
// 5 MA: operator=(Employee&& other)
std::cout << "\n Print:\n";
std::cout << " e1: " << e1.getName() << ", " << e1.getId() << '\n';
std::cout << " e2: " << e2.getName() << ", " << e2.getId() << '\n';
std::cout << " e3: " << e3.getName() << ", " << e3.getId() << '\n';
std::cout << " e4: " << e4.getName() << ", " << e4.getId() << '\n';
std::cout << "\n Fill and Print e4:\n";
e4.setName("Mario");
e4.setId("3-593-1");
std::cout << " e4: " << e4.getName() << ", " << e4.getId() << '\n';
std::cout << "\n=== SIMULATION COMPLETED ===\n";
// 6 De: ~Employee()
}
//================================================================================ END
/**
* ============================================================================
* File: Composite.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Composite Pattern. It treats single 
* components (CPU, RAM, SDD) and compositions (Motherboard, Case, Diskrack)
* uniformly through the 'Equipment' interface.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <vector>
//----------------------------------------------------------- Equipment (Component):
class Equipment
{
protected:
std::string name_;
public:
explicit Equipment(std::string name) : name_{std::move(name)} { }
virtual ~Equipment() = default;
virtual void print(int indentation = 0) const = 0;
};
//--------------------------------------------------------------------- Leaf Elements:
class CPU : public Equipment
{
public:
using Equipment::Equipment;
void print(int indentation = 0) const override
{
for (int i = 0; i < indentation; ++i) std::cout << "  ";
std::cout << "- CPU: " << name_ << std::endl;
}
};
class RAM : public Equipment
{
public:
using Equipment::Equipment;
void print(int indentation = 0) const override
{
for (int i = 0; i < indentation; ++i) std::cout << "  ";
std::cout << "- RAM: " << name_ << std::endl;
}
};
class SSD : public Equipment
{
public:
using Equipment::Equipment;
void print(int indentation = 0) const override
{
for (int i = 0; i < indentation; ++i) std::cout << "  ";
std::cout << "- SSD: " << name_ << std::endl;
}
};
//--------------------------------------------------------------------- Composite:
class Composite : public Equipment
// Is an Equipment that ...
{
private:
std::vector<std::unique_ptr<Equipment>> children_;
// has several Equipments
public:
using Equipment::Equipment;
void add(std::unique_ptr<Equipment> component)
{
children_.push_back(std::move(component));
}
void print(int indentation = 0) const override
{
for (int i = 0; i < indentation; ++i) std::cout << "  ";
std::cout << "+ Composite: " << name_ << std::endl;
for (const auto& child : children_)
{
child->print(indentation + 1);
}
}
};
//-------------------------------------------------------------------------- Main:
int main()
{
std::cout << "=== COMPUTER ASSEMBLY (COMPOSITE PATTERN) ===\n" << std::endl;
// 1. Create the main composite
auto mainBox = std::make_unique<Composite>("Tower Case");
// 2. Add simple leaf to main box
mainBox->add(std::make_unique<SSD>("Samsung 980 Pro"));
// 3. Create a motherboard (composite)
auto motherboard = std::make_unique<Composite>("ASUS ROG Motherboard");
// 4. Add leaves to motherboard
motherboard->add(std::make_unique<CPU>("Intel i9-13900K"));
motherboard->add(std::make_unique<RAM>("Corsair Vengeance 32GB"));
motherboard->add(std::make_unique<RAM>("Corsair Vengeance 32GB"));
// 5. Create a disk rack memory (composite)
auto diskrack = std::make_unique<Composite>("Disk Rack SDD Server Memory");
// 6. Add SSD memory into diskrack
diskrack->add(std::make_unique<SSD>("One Tera SSD disk"));
diskrack->add(std::make_unique<SSD>("One Tera SSD disk"));
diskrack->add(std::make_unique<SSD>("One Tera SSD disk"));
// 7. Add the diskrack (composite) into motherboard (composite)
motherboard->add(std::move(diskrack));
// 8. Add the motherboard (composite) into the box (composite)
mainBox->add(std::move(motherboard));
// 6. Print the whole structure uniformly starting at level 0
mainBox->print();
std::cout << "\n=== ASSEMBLY COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Decorator.cpp (Simple Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Decorator pattern with advanced string 
* formatting logic. It builds a natural description:
* 0 base:         "Coffe"
* 1 supplement:   "Coffee with Milk"
* 2 supplements:  "Coffee with Milk and Sugar"
* 3+ supplements: "Coffee with Milk, Sugar and Vanilla"
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
//--------------------------------------------------------- Base Interface:
class Decorated
{
public:
virtual ~Decorated() = default;
virtual std::string getDescription() const = 0;
virtual double getCost() const = 0;
};
//----------------------------------------------------- Concrete Decorated:
class Coffee : public Decorated
{
public:
~Coffee() override
{
std::cout << " [Cleanup] Coffee base object destroyed.\n";
}
std::string getDescription() const override
{
return "Coffee";
}
double getCost() const override
{
return 2.0; 
}
};
//-------------------------------------------------------- Base Decorator:
class Decorator : public Decorated
// "is a" Decorated
{
protected:
std::unique_ptr<Decorated> decorated_;
//  and "has a" Decorated
// Smart logic to build a natural English sentence
std::string wrapDescription(const std::string& ingredient) const
{
std::string current = decorated_->getDescription();
size_t andPos = current.find(" and ");
size_t withPos = current.find(" with ");
// Case 1: We already have an "and" (e.g., "Coffee with Milk and Sugar")
// Replace the old " and " with ", " and add the new " and ingredient"
if (andPos != std::string::npos)
{
current.replace(andPos, 5, ", ");
return current + " and " + ingredient;
}
// Case 2: We have a "with" but no "and" (e.g., "Coffee with Milk")
// Just add " and ingredient"
if (withPos != std::string::npos)
{
return current + " and " + ingredient;
}
// Case 3: No supplements yet (e.g., "Coffee")
// Add the first " with ingredient"
return current + " with " + ingredient;
}
public:
explicit Decorator(std::unique_ptr<Decorated> target)
: decorated_{std::move(target)} { }
~Decorator() override
{
std::cout << " [Cleanup] Ingredient wrapper destroyed.\n";
}
std::string getDescription() const override
{
return decorated_->getDescription();
}
double getCost() const override
{
return decorated_->getCost();
}
};
//----------------------------------------------------- Concrete Decorator A:
class Milk : public Decorator
{
public:
using Decorator::Decorator;
std::string getDescription() const override
{
return wrapDescription("Milk");
}
double getCost() const override
{
return decorated_->getCost() + 0.5;
}
};
//----------------------------------------------------- Concrete Decorator B:
class Sugar : public Decorator
{
public:
using Decorator::Decorator;
std::string getDescription() const override
{
return wrapDescription("Sugar");
}
double getCost() const override
{
return decorated_->getCost() + 0.2;
}
};
//----------------------------------------------------- Concrete Decorator C:
class Vanilla : public Decorator
{
public:
using Decorator::Decorator;
std::string getDescription() const override
{
return wrapDescription("Vanilla");
}
double getCost() const override
{
return decorated_->getCost() + 0.7;
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== COFFEE SHOP (SIMPLE DECORATOR) ===\n" << std::endl;
// We build a Coffee with Milk, Sugar and Vanilla.
// --- Step-by-step ordering (Dynamic Decoration) ---
// 1. Start with a basic Coffee
std::unique_ptr<Decorated> myDrink = std::make_unique<Coffee>();
std::cout << " Order: " << myDrink->getDescription();
std::cout << ", Cost: $" << myDrink->getCost() << std::endl;
// 2. Add Milk
myDrink = std::make_unique<Milk>(std::move(myDrink));
std::cout << " Order: " << myDrink->getDescription();
std::cout << ", Cost: $" << myDrink->getCost() << std::endl;
// 3. Add Sugar
myDrink = std::make_unique<Sugar>(std::move(myDrink));
std::cout << " Order: " << myDrink->getDescription();
std::cout << ", Cost: $" << myDrink->getCost() << std::endl;
// 4. Add Vanilla
myDrink = std::make_unique<Vanilla>(std::move(myDrink));
std::cout << " Order: " << myDrink->getDescription();
std::cout << ", Cost: $" << myDrink->getCost() << std::endl;
std::cout << "\n--- Closing the shop. Cleaning up orders... ---\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Decorator.cpp (Advanced Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This advanced implementation of the Decorator pattern simulates a dynamic
* beverage ordering system. 
* 
* Features:
* 1. Base components: Coffee and Tea.
* 2. Parameterized Decorators: Milk (with types), Sugar & Vanilla (portions).
* 3. State Decorator: Temperature (Hot/Iced/ExtraHot). 
* 4. Automatic Grammar: Logic to handle "with", "and", and commas correctly.
* 5. Memory Management: std::unique_ptr ensures no leaks.
* 6. Configuration: Use #define SHOW_CLEANUP to toggle destructor messages.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
// Toggle this definition to show or hide memory cleanup messages
//#define SHOW_CLEANUP
//--------------------------------------------------------- Base Interface:
class Decorated
{
public:
virtual ~Decorated() = default;
virtual std::string getDescription() const = 0;
virtual double getCost() const = 0;
};
//----------------------------------------------------- Concrete Decorated A:
class Coffee : public Decorated
{
public:
~Coffee() override 
{ 
#ifdef SHOW_CLEANUP
std::cout << " [Cleanup] Coffee base destroyed.\n"; 
#endif
}
std::string getDescription() const override { return "Coffee"; }
double getCost() const override { return 2.0; }
};
//----------------------------------------------------- Concrete Decorated B:
class Tea : public Decorated
{
public:
~Tea() override 
{ 
#ifdef SHOW_CLEANUP
std::cout << " [Cleanup] Tea base destroyed.\n"; 
#endif
}
std::string getDescription() const override { return "Tea"; }
double getCost() const override { return 1.5; }
};
//-------------------------------------------------------- Base Decorator:
class Decorator : public Decorated
// "is a" Decorated and ...
{
protected:
std::unique_ptr<Decorated> decorated_;
// "has a" Decorated
// Helper for ingredients list (with / and / ,)
std::string addIngredient(const std::string& ingredient) const
{
std::string current = decorated_->getDescription();
size_t andPos = current.find(" and ");
size_t withPos = current.find(" with ");
if (andPos != std::string::npos)
{
current.replace(andPos, 5, ", ");
return current + " and " + ingredient;
}
if (withPos != std::string::npos)
{
return current + " and " + ingredient;
}
return current + " with " + ingredient;
}
public:
explicit Decorator(std::unique_ptr<Decorated> target)
: decorated_{std::move(target)} { }
~Decorator() override 
{ 
#ifdef SHOW_CLEANUP
std::cout << " [Cleanup] Decorator wrapper destroyed.\n"; 
#endif
}
std::string getDescription() const override { return decorated_->getDescription(); }
double getCost() const override { return decorated_->getCost(); }
};
//----------------------------------------------------- Milk Decorator:
enum class MilkType { Whole, Soy, Almond };
class Milk : public Decorator
{
private:
MilkType type_;
public:
Milk(std::unique_ptr<Decorated> target, MilkType type)
: Decorator{std::move(target)}, type_{type} { }
std::string getDescription() const override
{
std::string name;
switch (type_)
{
case MilkType::Soy:    name = "Soy Milk";    break;
case MilkType::Almond: name = "Almond Milk"; break;
default:               name = "Milk";        break;
}
return addIngredient(name);
}
double getCost() const override
{
double extra = (type_ == MilkType::Almond) ? 0.9 : 0.5;
return decorated_->getCost() + extra;
}
};
//----------------------------------------------------- Sugar Decorator:
class Sugar : public Decorator
{
private:
int portions_;
public:
Sugar(std::unique_ptr<Decorated> target, int portions)
: Decorator{std::move(target)}, portions_{portions} { }
std::string getDescription() const override
{
std::string label = (portions_ > 1) ? " portions of Sugar" : " portion of Sugar";
return addIngredient(std::to_string(portions_) + label);
}
double getCost() const override
{
return decorated_->getCost() + (portions_ * 0.15);
}
};
//----------------------------------------------------- Vanilla Decorator:
class Vanilla : public Decorator
{
private:
int portions_;
public:
Vanilla(std::unique_ptr<Decorated> target, int portions)
: Decorator{std::move(target)}, portions_{portions} { }
std::string getDescription() const override
{
std::string label = (portions_ > 1) ? " portions of Vanilla" : " portion of Vanilla";
return addIngredient(std::to_string(portions_) + label);
}
double getCost() const override
{
return decorated_->getCost() + (portions_ * 0.40);
}
};
//----------------------------------------------------- Temperature Decorator:
enum class Temp { Hot, Iced, ExtraHot };
class Temperature : public Decorator
{
private:
Temp temp_;
public:
Temperature(std::unique_ptr<Decorated> target, Temp temp)
: Decorator{std::move(target)}, temp_{temp} { }
std::string getDescription() const override
{
std::string prefix;
switch (temp_)
{
case Temp::Iced:     prefix = "Iced ";      break;
case Temp::ExtraHot: prefix = "Extra Hot "; break;
default:             prefix = "Hot ";       break;
}
return prefix + decorated_->getDescription();
}
/* Temperature doesn't need to decorete getCost if Temperature is free.
double getCost() const override
{
// Iced has a surcharge, Hot and Extra Hot are free
double extra = (temp_ == Temp::Iced) ? 0.35 : 0.0;
return decorated_->getCost() + extra;
}
*/
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== ADVANCED COFFEE SHOP (DYNAMIC DECORATORS) ===\n" << std::endl;
std::unique_ptr<Decorated> drink;
std::cout << "\n------------------------------------------\n" << std::endl;
// --- ORDER 1: Complex Iced Coffee ---
std::cout << "--- Order 1 ---\n";
drink = std::make_unique<Coffee>(); 
drink = std::make_unique<Milk>(std::move(drink), MilkType::Almond);
drink = std::make_unique<Vanilla>(std::move(drink), 2);
drink = std::make_unique<Temperature>(std::move(drink), Temp::Iced);
std::cout << " Receipt: " << drink->getDescription() << std::endl;
std::cout << " Total Cost: $" << drink->getCost() << std::endl;
std::cout << "\n------------------------------------------\n" << std::endl;
// --- ORDER 2: Simple Hot Tea ---
std::cout << "--- Order 2 ---\n";
drink = std::make_unique<Tea>();
drink = std::make_unique<Sugar>(std::move(drink), 1);
drink = std::make_unique<Temperature>(std::move(drink), Temp::Hot);
std::cout << " Receipt: " << drink->getDescription() << std::endl;
std::cout << " Total Cost: $" << drink->getCost() << std::endl;
std::cout << "\n------------------------------------------\n" << std::endl;
// --- ORDER 3: Loaded Soy Coffee ---
std::cout << "--- Order 3 ---\n";
drink = std::make_unique<Coffee>();
drink = std::make_unique<Milk>(std::move(drink), MilkType::Soy);
drink = std::make_unique<Sugar>(std::move(drink), 3);
drink = std::make_unique<Vanilla>(std::move(drink), 1);
drink = std::make_unique<Temperature>(std::move(drink), Temp::Hot);
std::cout << " Receipt: " << drink->getDescription() << std::endl;
std::cout << " Total Cost: $" << drink->getCost() << std::endl;
std::cout << "\n------------------------------------------\n" << std::endl;
// --- ORDER 4: Loaded Soy Coffee (Increased Temperature) ---
std::cout << "--- Order 4 (Same as 3 but Extra Hot) ---\n";
drink = std::make_unique<Coffee>();
drink = std::make_unique<Milk>(std::move(drink), MilkType::Soy);
drink = std::make_unique<Sugar>(std::move(drink), 3);
drink = std::make_unique<Vanilla>(std::move(drink), 1);
drink = std::make_unique<Temperature>(std::move(drink), Temp::ExtraHot);
std::cout << " Receipt: " << drink->getDescription() << std::endl;
std::cout << " Total Cost: $" << drink->getCost() << std::endl;
#ifdef SHOW_CLEANUP
std::cout << "\n--- Closing Shop. Objects will be destroyed below ---" << std::endl;
#endif
std::cout << "\n------------------------------------------\n" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Static_Mixin_Decorator.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Static Decorator pattern, also known as 
* "Mixin Inheritance". Unlike the Dynamic versions (Simple/Advanced), 
* the composition is defined at compile-time using C++ Templates.
* 
* --- RELATIONSHIP NOTE:
* In the Dynamic Decorator, the relationship is 'is a' AND 'has a' (composition).
* In the Static Decorator, the relationship is strictly 'is a' (inheritance).
* The static decorator "is a" Decorated, but it doesn't "have a" Decorated; it wraps the 
* functionality by becoming part of the class hierarchy itself, eliminating 
* the need for internal pointers.
* 
* --- THE PROBLEM:
* In an industrial coffee machine factory, we design specialized machines 
* that only produce one specific recipe (e.g., "The Sweet Latte Machine"). 
* Since the recipe never changes during the machine's life, we don't need 
* the flexibility of pointers or virtual tables. We need maximum performance 
* and a small memory footprint.
* 
* --- THE SOLUTION:
* We use templates where decorators inherit from their template argument:
* 'class Milk : public Decorated'. This allows the compiler to "flatten" the 
* inheritance chain, enabling aggressive optimizations like inlining.
* This implementation also ensures that the natural language description 
* ("with", "and", ",") is maintained with zero runtime overhead.
*
* Note:
* Since we don't have a common 'Decorator' base class with a pointer, 
* each template class handles the formatting logic independently to 
* maintain the "Zero-overhead" principle.
* ============================================================================
*/
#include <iostream>
#include <string>
//--------------------------------------------------------- Base Component:
class Coffee_ToBeDecorated
{
public:
std::string getDescription() const
{
return "Coffee";
}
double getCost() const
{
return 2.0;
}
protected:
// Centralized formatting description logic
std::string formatDescription(std::string current, const std::string& ingredient) const
{
size_t andPos = current.find(" and ");
if(andPos != std::string::npos)
{
current.replace(andPos, 5, ", ");
return current + " and " + ingredient;
}
if(current.find(" with ") != std::string::npos)
{
return current + " and " + ingredient;
}
return current + " with " + ingredient;
}
};
//--------------------------------------------------------- Static Decorator A:
template <typename Decorated>
class Milk : public Decorated
// Static Decorator "is a" Decorated,
{
// but doesn't "have a" Decorated
public:
std::string getDescription() const
{
return Decorated::formatDescription(Decorated::getDescription(), "Milk");
}
double getCost() const
{
return Decorated::getCost() + 0.5;
}
};
//--------------------------------------------------------- Static Decorator B:
template <typename Decorated>
class Sugar : public Decorated
// Static Decorator "is a" Decorated,
{
// but doesn't "have a" Decorated
public:
std::string getDescription() const
{
return Decorated::formatDescription(Decorated::getDescription(), "Sugar");
}
double getCost() const
{
return Decorated::getCost() + 0.2;
}
};
//--------------------------------------------------------- Static Decorator C:
template <typename Decorated>
class Vanilla : public Decorated
// Static Decorator "is a" Decorated,
{
// but doesn't "have a" Decorated
public:
std::string getDescription() const
{
return Decorated::formatDescription(Decorated::getDescription(), "Vanilla");
}
double getCost() const
{
return Decorated::getCost() + 0.7;
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== INDUSTRIAL COFFEE FACTORY (STATIC DECORATORS) ===\n" << std::endl;
// Recipe 1: Basic Latte (Milk<Coffee_ToBeDecorated>)
Milk<Coffee_ToBeDecorated> latte;
std::cout << " Recipe 1 (Latte): " << latte.getDescription() << std::endl;
std::cout << " Static Cost: $" << latte.getCost() << std::endl;
// Recipe 2: Sweet Latte (Sugar<Milk<Coffee_ToBeDecorated>>)
Sugar<Milk<Coffee_ToBeDecorated>> sweetLatte;
std::cout << "\n Recipe 2 (Sweet Latte): " << sweetLatte.getDescription() << std::endl;
std::cout << " Static Cost: $" << sweetLatte.getCost() << std::endl;
// Recipe 3: Vanilla Dream (Vanilla<Sugar<Milk<Coffee_ToBeDecorated>>>)
// Now this will correctly print: "Coffee with Milk, Sugar and Vanilla"
Vanilla<Sugar<Milk<Coffee_ToBeDecorated>>> vanillaDream;
std::cout << "\n Recipe 3 (Vanilla Dream): " << vanillaDream.getDescription() << std::endl;
std::cout << " Static Cost: $" << vanillaDream.getCost() << std::endl;
std::cout << "\n--- Production lines are optimized. No pointers used. ---\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Static_Deducing_this_Decorator.cpp (Modern C++23 Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program implements a cutting-edge version of the Static Decorator pattern 
* (Mixin Inheritance) using C++23 "Deducing This". It models a professional 
* Industrial Sensor Analytics system where raw hardware readings are enhanced 
* through a chain of compile-time Mixins (Averaging and Alarms).
* 
* --- THE ARCHITECTURAL PROBLEM (INTERFACE LOSS):
* Traditional Static Decorators (see Example 03) suffer from "Interface Loss". 
* If a method defined in the deepest Base class is invoked, it typically 
* returns a reference to the Base class. This "breaks the chain" because the 
* compiler "forgets" about the outer decorators, preventing the client from 
* accessing decorator-specific methods in the same fluent expression.
* 
* --- THE C++23 SOLUTION (CALIBRATABLE STATIC MIXINS):
* By using 'this auto&& self' in both the Base class and every Decorator, we 
* achieve "Interface Persistence". The 'self' parameter captures the 
* "most derived" type (the outermost decorator) at the call site. This allows 
* for a perfect, perfectly-typed Fluent Interface where methods from 
* different layers can be interleaved seamlessly.
* 
* Furthermore, this architecture enables "Runtime Calibration": the structure 
* is fixed at compile-time for maximum performance, but functional parameters 
* (like alarm limits or filter windows) remain adjustable at runtime without 
* re-instantiating the object.
* 
* --- TECHNICAL MECHANICS:
* 1. Zero-Overhead Polymorphism: No VTables or pointers are used. The compiler 
*    flattens the inheritance chain, enabling total inlining of the 
*    processing logic and zero runtime indirection.
* 2. Perfect Forwarding: Every setter returns 'std::forward<decltype(self)>(self)', 
*    ensuring the value category (lvalue/rvalue) is preserved across the chain.
* 3. Functional Enhancement: The decorators transform behavior (e.g., from 
*    point readings to moving averages) via static dispatch.
* 4. Deterministic Simulation: The base sensor uses a fixed data array to 
*    ensure predictable and testable hardware readings.
* 5. Total Hierarchy Visibility: Since 'self' always refers to the "most 
*    derived" object, every layer (including the Base class) has full access 
*    to the methods of all other layers. This creates a transparent 
*    composition where the order of template nesting does not affect 
*    functional behaviorprovided method names are uniqueas every component 
*    of the composite object is visible to the entire hierarchy through the 
*    explicit object parameter.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <array>
#include <utility>
// Required for std::forward
//--------------------------------------------------------- Base Component:
/**
* AnalogSensor: Models the physical hardware.
* It contains a simulated buffer of raw voltage readings.
*/
class AnalogSensor
{
private:
int sensorID_{0};
size_t readIndex_{0};
// Simulated deterministic hardware readings (Voltages)
// In real life, these values are not deterministic.
static constexpr std::array<double, 16> simulatedData_
{
1.2, 1.5, 2.8, 2.5, 1.1, 0.9, 3.2, 3.5, 1.8, 1.4, 2.1, 3.2, 1.5, 1.1, 0.2, 0.5,
};
public:
// C++23: Every method in the base returns the outermost decorator!
auto&& setID(this auto&& self, int id) noexcept
{
// Mandatory: 'self' is required because Deducing This suppresses
// the implicit 'this' pointer and class-member lookup.
self.sensorID_ = id;
return std::forward<decltype(self)>(self);
}
/**
* Returns the next raw value from the hardware buffer.
*/
double readRaw()
{
double val = simulatedData_[readIndex_];
readIndex_ = (readIndex_ + 1) % simulatedData_.size();
// Wrap around
return val;
}
int getID() const { return sensorID_; }
};
//-------------------------------------------------- Static Decorator Averager:
/**
* Averager Decorator:
* Injects a Moving Average filter into the sensor pipeline.
* It intercepts the reading logic to provide a smoothed value
* based on a configurable window size.
*/
template <typename Decorated>
class Averager : public Decorated
{
private:
int windowSize_{1};
public:
// Runtime Calibration: Window size can be adjusted on the fly
auto&& setWindowSize(this auto&& self, int size) noexcept
{
// Mandatory: 'self' is required because Deducing This suppresses
// the implicit 'this' pointer and class-member lookup.
self.windowSize_ = (size > 10) ? 10 : (size < 1) ? 1 : size;
// Constraint check
return std::forward<decltype(self)>(self);
}
/**
* Performs the average calculation by pulling multiple raw values 
* from the underlying sensor logic.
*/
double read(this auto&& self)
{
double sum = 0.0;
for(int i = 0; i < self.windowSize_; ++i) sum += self.readRaw(); 
double average = sum / self.windowSize_;
std::cout << " [Filter] Averaged " << self.windowSize_ 
<< " samples. Result: " << average << "V\n";
return average;
}
};
//-------------------------------------------- Static Decorator ThresholdAlarm:
/**
* ThresholdAlarm Decorator:
* Adds safety monitoring to the sensor pipeline.  It checks
* if the processed sensor value exceeds a defined safety
* limit.
*/
template <typename Decorated>
class ThresholdAlarm : public Decorated
{
private:
double limit_{5.0};
public:
// Runtime Calibration: Alarm limits are dynamic
auto&& setAlarmLimit(this auto&& self, double limit) noexcept
{
// Mandatory: 'self' is required because Deducing This suppresses
// the implicit 'this' pointer and class-member lookup.
self.limit_ = limit;
return std::forward<decltype(self)>(self);
}
/**
* High-level logic that coordinates the reading and the safety check.
*/
void monitor(this auto&& self)
{
double currentVal = self.read();
// Static dispatch to Averager
if (currentVal > self.limit_)
{
std::cout << " [ALARM] Sensor " << self.getID() 
<< " reporting " << currentVal << "V. EXCEEDS LIMIT (" 
<< self.limit_ << "V)!\n";
}
else
{
std::cout << " [Monitor] Sensor " << self.getID() 
<< " is stable at " << currentVal << "V.\n";
}
}
};
//------------------------------------------------------------------- Main:
int main()
{
std::cout << "=== INDUSTRIAL SENSOR ANALYTICS (CALIBRATABLE STATIC DECORATORS) ===\n"
<< std::endl;
/**
* CONSTRUCTION OF THE SMART SENSOR:
* We wrap the AnalogSensor with an Averager, then with a ThresholdAlarm.
* 
* THE POWER OF DEDUCING THIS:
* Notice how we interleave calls from different hierarchy levels:
* - .setID (Base)
* - .setWindowSize (Decorator A)
* - .setAlarmLimit (Decorator B)
*/
std::cout << "--- PHASE 1: Initialization & Calibration ---\n";
/**
* ARCHITECTURAL NOTE ON COMPOSITION ORDER:
* Due to C++23 "Deducing This", the order of template nesting is functionally 
* agnostic in this specific implementation (e.g., Averager<ThresholdAlarm<...>> 
* would yield the same result as ThresholdAlarm<Averager<...>>). 
*
* Since 'self' always represents the most derived (outermost) type, every 
* layer has total visibility of the entire hierarchy's API. This transforms 
* the "Onion-Layer" Decorator into a "Transparent Composite" where any part 
* can invoke any other part seamlessly.
*/
// auto smartSensor = Averager<ThresholdAlarm<AnalogSensor>>{}
auto smartSensor = ThresholdAlarm<Averager<AnalogSensor>>{}
.setID(101)
// Defined in AnalogSensor (Base)
.setWindowSize(4)
// Defined in Averager (Decorator A)
.setAlarmLimit(2.2);
// Defined in ThresholdAlarm (Decorator B)
std::cout << " Sensor " << smartSensor.getID() << " configured.\n";
std::cout << "\n--- PHASE 2: High Sensitivity Cycle ---\n";
smartSensor.monitor();
std::cout << "\n Taking another reading of data:\n";
smartSensor.monitor();
std::cout << "\n--- PHASE 3: Hot Re-Calibration (Values defined at execution time) ---\n";
// We adjust the filter and the safety limit in a single fluent chain
smartSensor.setWindowSize(3)
.setAlarmLimit(1.8)
.monitor();
std::cout << "\n--- PHASE 4: Identity Preservation ---\n";
// Even changing the ID doesn't break the decorator's API
smartSensor.setID(292).setAlarmLimit(1.9).monitor();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Facade.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Facade pattern, which provides a simplified 
* interface to a complex system. The goal is to hide the complexity 
* of booting a computer from the client.
* ============================================================================
*/
#include <iostream>
#include <memory>
//--------------------------------------------------------- Subsystem Classes:
// These classes represent the complex internals of the computer.
class CPU
{
public:
void freeze() const { std::cout << " [CPU] Freezing CPU...\n"; }
void jump(long position) const { std::cout << " [CPU] Jumping to address: " << position << "\n"; }
void execute() const { std::cout << " [CPU] Executing instructions...\n"; }
};
class HardDrive
{
public:
const char* read(long lba, int size) const
{
static constexpr char data[] = "System loaded from disk";
std::cout << " [HardDrive] Reading " << size << " bytes from sector " << lba << "...\n";
return data;
}
};
class Memory
{
public:
void load(long position, const char* data) const
{
std::cout << " [Memory] Loading '" << data << "' into memory at " << position << "...\n";
}
};
//--------------------------------------------------------- Facade Class:
// The Facade provides a simplified interface to the complex subsystem.
class ComputerFacade
{
private:
// The Facade *HAS A* CPU, Memory, and HardDrive.
std::unique_ptr<CPU>       cpu_        = std::make_unique<CPU>();
std::unique_ptr<Memory>    memory_     = std::make_unique<Memory>();
std::unique_ptr<HardDrive> hard_drive_ = std::make_unique<HardDrive>();
public:
void start() const
{
std::cout << "=== BOOTING COMPUTER ===\n";
const long kBootAddress = 300;
const long kBootSector  = 49;
const int  kSectorSize  = 1024;
cpu_->freeze();
memory_->load(kBootAddress, hard_drive_->read(kBootSector, kSectorSize));
cpu_->jump(kBootAddress);
cpu_->execute();
std::cout << "\n=== COMPUTER BOOTED SUCCESSFULLY! ===\n";
}
};
//--------------------------------------------------------- Client Code (main):
int main()
{
std::cout << "=== FACADE PATTERN SIMULATION ===\n" << std::endl;
ComputerFacade computer;
computer.start();
std::cout << "\n=== END OF DEMONSTRATION ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Flyweight.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Flyweight pattern. We separate the intrinsic 
* state (Model) from the extrinsic state (Airplane).
* 
* Simulation:
* We create a large number of 'Airplane' instances to show that memory usage 
* is minimized: even with thousands of airplanes, only 3 'Model' objects 
* are stored in memory.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <random>
// Essential for modern random logic
// Toggle this definition to show or hide memory cleanup messages
#define SHOW_CLEANUP
//--------------------------------------------------------- Intrinsic State:
class Model
{
private:
std::string name_;
int         capacity_;
int         speed_;
std::string deployDate_;
public:
Model(std::string name, int capacity, int speed, std::string date)
: name_{std::move(name)}, capacity_{capacity},
speed_{speed}, deployDate_{std::move(date)} { }
~Model()
{
#ifdef SHOW_CLEANUP
std::cout << " [~Model] Model '" << name_ << "' destroyed.\n";
#endif
}
void showDetails() const
{
std::cout << " Name: "         << name_ 
<< ", Capacity: "    << capacity_    << " passengers"
<< ", Speed: "       << speed_       << " knots"
<< ", Deploy date: " << deployDate_;
}
std::string getName()
{
return name_;
}
};
//--------------------------------------------------------- Flyweight Factory:
class FlyweightFactory
{
private:
std::unordered_map<int, std::shared_ptr<Model>> modelCache_;
public:
std::shared_ptr<Model> getModel(int type)
{
if(modelCache_.find(type) == modelCache_.end())
{
std::cout << " [Factory] Creating new model type: " << type << "\n";
switch(type)
{
case 747: modelCache_[type] = std::make_shared<Model>("Boeing 747", 467, 495, "Sep 1968"); break;
case 380: modelCache_[type] = std::make_shared<Model>("Airbus 380", 545, 510, "Apr 2005"); break;
case 787: modelCache_[type] = std::make_shared<Model>("Boeing 787", 330, 488, "Dec 2009"); break;
case 220: modelCache_[type] = std::make_shared<Model>("Airbus 220", 150, 450, "Sep 2013"); break;
default:  throw std::invalid_argument("Unknown aeroplane type");
}
}
return modelCache_[type];
}
};
//--------------------------------------------------------- Extrinsic State:
class Airplane
{
private:
std::shared_ptr<Model> model_;
// Almost everything is stored here.
int id_;
// Only a few unique values are stored here
public:
Airplane(std::shared_ptr<Model> model, int id) : model_{std::move(model)}, id_{id} { }
~Airplane()
{
#ifdef SHOW_CLEANUP
std::cout << " [~Airplane] Airplane " << id_ << " model " << model_->getName() << " destroyed.\n";
#endif
}
void showDetails() const
{
model_->showDetails();
std::cout << ", Serial No: " << id_ << '\n';
}
};
int getRandomModelType()
{
static const int modelTypes[] = {747, 380, 787, 220};
static std::random_device rd; 
static std::mt19937 gen(rd()); 
static std::uniform_int_distribution<int> dist(0, 3);
// Pick a random index and return the model type
return modelTypes[dist(gen)];
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== FLYWEIGHT PATTERN (MASSIVE SCALING SIMULATION) ===\n" << std::endl;
FlyweightFactory factory;
std::vector<Airplane> fleet;
// Create a large fleet of airplanes
const int numAirplanes = 100;
fleet.reserve(numAirplanes);
std::cout << " [Factory] Creating " << numAirplanes << " airplanes..." << std::endl;
for (int i = 0; i < numAirplanes; ++i)
{
// Choose random types from 747, 380, 787 and 220.
int type = getRandomModelType();
fleet.emplace_back(factory.getModel(type), i+1);
}
std::cout << " [Factory] Total fleet created." << std::endl;
std::cout << " Fleet size: " << fleet.size() << std::endl;
// Check details of the first and last one to verify it works
std::cout << "\n First airplane details:\n ";
fleet.front().showDetails();
std::cout << "\n Last airplane details:\n ";
fleet.back().showDetails();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Proxy.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Proxy pattern. The 'ProxyCar' acts as an 
* interface for 'Car'. It controls access to the real object by checking 
* the driver's age before allowing the call to proceed.
* 
* --- KEY FEATURES:
* - Protection Proxy: Controls access based on conditions (age).
* - Lazy Initialization: The real 'Car' is only created when needed.
* - Uniform Interface: The client interacts with 'ICar' without knowing 
*   if it is talking to a Proxy or the real object.
* ============================================================================
*/
#include <iostream>
#include <memory>
//--------------------------------------------------------- Subject Interface:
class ICar
{
public:
virtual ~ICar() = default;
virtual void drive() const = 0;
};
//------------------------------------------------ Car (real object):
class Car : public ICar
{
public:
Car() { std::cout << " [System] Real Car created.\n"; }
~Car() override { std::cout << " [Cleanup] Real Car destroyed.\n"; }
void drive() const override 
{ 
std::cout << " -> Driving the car!\n"; 
}
};
//--------------------------------------------------------- ProxyCar:
class ProxyCar : public ICar
{
private:
mutable std::unique_ptr<Car> realCar_;
// Lazy initialized
int driverAge_;
public:
explicit ProxyCar(int driverAge) : driverAge_{driverAge} { }
void drive() const override
{
if (driverAge_ > 16)
{
// Lazy Initialization: Create the real object only when needed
if (!realCar_) realCar_ = std::make_unique<Car>();
realCar_->drive();
}
else
std::cout << " [Proxy] Access denied: Driver is too young.\n";
}
};
//--------------------------------------------------------- Main:
int main()
{
std::cout << "=== PROXY PATTERN SIMULATION ===\n" << std::endl;
std::cout << "--- Attempt 1: Young driver ---\n";
std::unique_ptr<ICar> proxy1 = std::make_unique<ProxyCar>(16);
proxy1->drive();
std::cout << "\n--- Attempt 2: Adult driver ---\n";
std::unique_ptr<ICar> proxy2 = std::make_unique<ProxyCar>(25);
proxy2->drive();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Mixin.cpp (Simple Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates "Horizontal Composition" through Mixin 
* Inheritance. Using Variadic Templates, we create a generic 'Entity' 
* that can inherit multiple capabilities (Mixins) at the same time.
* 
* Unlike the Decorator pattern (which wraps objects), the Mixin pattern 
* builds a single, flat object containing all the desired features.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <utility>
// std::move
//---------------------------------------------------- Features (Mixin Parts):
//---------------------------------- Laser:
class Laser
{
private:
int intensity_{2};
public:
void laser_fire()
{
std::cout << "\tLaser fire intensity " << intensity_ << "\n";
if(intensity_ > 2) --intensity_;
}
void laser_set_intensity(int i) 
{
if(i < 2) i = 2; 
intensity_ = i;
}
};
//----------------------------------- Walk:
class Walk
{
private:
int speed_{0};
public:
void walk() const
{
std::cout << "\tWalk speed " << speed_ << "\n";
}
void walk_set_speed(int s) 
{
speed_ = s;
}
};
//------------------------------------ Gun:
class Gun
{
private:
int bullets_{0};
public:
void gun_fire()
{
if(bullets_ > 0)
{
std::cout << "\tGun fire\n"; 
--bullets_;
}
else 
{
std::cout << "\tGun no more bullets\n";
}
}
void gun_set_bullets(int b) 
{
bullets_ = b;
}
};
//------------------------------------ Fly:
class Fly
{
private:
int fuel_{0};
public:
void fly()
{
if(fuel_ > 0)
{
std::cout << "\tFlying\n";
--fuel_;
}
else
{
std::cout << "\tNo fuel to fly\n";
}
}
void load_fuel(int f) 
{
fuel_ = f;
}
};
//--------------------------------------------------------------------- Entity:
template<class ... Mixins>
// Inheriting from a pack of templates
class Entity : public Mixins...
{
private:
std::string name_;
public:
explicit Entity(std::string n) : name_{std::move(n)} { }
void print_name() const 
{
std::cout << "\n" << name_ << ":\n";
}
};
//------------------------------------------------------------ Mixin Entities:
// Using aliases to define new types by combining mixins
using Dragon = Entity<Fly, Laser>;
Dragon createDragon(std::string name, int fuel, int intensity)
{
Dragon d{std::move(name)};
d.load_fuel(fuel);
d.laser_set_intensity(intensity);
return d;
}
using Elephant = Entity<Walk, Gun>;
Elephant createElephant(std::string name, int speed, int bullets)
{
Elephant e{std::move(name)};
e.walk_set_speed(speed);
e.gun_set_bullets(bullets);
return e;
}
using Airplane = Entity<Fly, Laser, Gun>;
Airplane createAirplane(std::string name, int fuel, int intensity, int bullets)
{
Airplane a{std::move(name)};
a.load_fuel(fuel);
a.laser_set_intensity(intensity);
a.gun_set_bullets(bullets);
return a;
}
//------------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MIXIN PATTERN SIMULATION (HORIZONTAL COMPOSITION) ===\n";
//---------------------------- Dragon
Dragon dragon = createDragon("Dragon", 2, 4);
dragon.print_name();
dragon.fly();
dragon.laser_fire();
dragon.fly();
dragon.fly();
std::cout << "\tLoading fuel 1\n";
dragon.load_fuel(1);
dragon.fly();
dragon.fly();
dragon.laser_fire();
//----------------------------- Elephant
Elephant elephant = createElephant("Elephant", 3, 2);
elephant.print_name();
elephant.walk();
elephant.gun_fire();
elephant.gun_fire();
elephant.gun_fire();
//----------------------------- Airplane
Airplane airplane = createAirplane("Airplane", 2, 3, 2);
airplane.print_name();
airplane.fly();
airplane.laser_fire();
airplane.gun_fire();
airplane.gun_fire();
airplane.fly();
airplane.laser_fire();
airplane.gun_fire();
airplane.fly();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Mixin.cpp (Advanced Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This advanced implementation demonstrates two powerful C++ techniques:
* 1. Mixin Inter-dependency: The 'Fly' mixin requires the 'Tank' mixin 
*    to function, showing how components can interact within the same entity.
* 2. Static Reflection/Visitation: Using Fold Expressions (C++17) to 
*    iterate over all inherited base classes (Mixins) at compile-time.
* 
* This approach provides a high-performance alternative to the traditional 
* Visitor pattern, avoiding virtual tables and dynamic dispatch.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <utility>
//----------------------------------------------------------- Features (Mixins):
//------------------------------------------------------ Laser:
class Laser
{
private:
int intensity_{2};
public:
void laser_fire()
{
std::cout << "\tLaser fire intensity " << intensity_ << "\n";
if(intensity_ > 2) --intensity_;
}
void laser_set_intensity(int i) 
{
if(i < 2) i = 2; 
intensity_ = i;
}
void print() const 
{
std::cout << "  - Laser with intensity " << intensity_ << "\n";
}
};
//------------------------------------------------------- Walk:
class Walk
{
private:
int speed_{0};
public:
void walk() const
{
std::cout << "\tWalk speed " << speed_ << "\n";
}
void walk_set_speed(int s) 
{
speed_ = s;
}
void print() const 
{
std::cout << "  - Walk with speed " << speed_ << "\n";
}
};
//-------------------------------------------------------- Gun:
class Gun
{
private:
int bullets_{0};
public:
void gun_fire()
{
if(bullets_ > 0) 
{
std::cout << "\tGun fire\n"; 
--bullets_;
}
else
{
std::cout << "\tGun no more bullets\n";
}
}
void gun_set_bullets(int b) 
{
bullets_ = b;
}
void print() const 
{
std::cout << "  - Gun with " << bullets_ << " bullets\n";
}
};
//------------------------------------------------------- Tank:
class Tank
{
private:
int fuel_{0};
public:
void tank_load_fuel(int f) 
{
fuel_ = f;
}
bool tank_get_fuel(int quantity)
{
if(fuel_ < quantity) return false;
fuel_ -= quantity;
return true;
}
void print() const 
{
std::cout << "  - Tank with " << fuel_ << " of fuel\n";
}
};
//-------------------------------------------------------- Fly:
class Fly
{
private:
int altitude_{300};
int speed_{40};
public:
void fly_set_altitude_speed(int altitude, int speed)
{
altitude_= altitude;
speed_ = speed;
}
// Alternative 1: Rigid dependency: Only class Tank (o a derived class) can be used to call fly.
// void fly(Tank& entity)
// {
// Alternative 2: Generic template (The Entity type must be explicitly defined)
// Any class that provides 'tank_get_fuel' is accepted at compilation time.
// The Entity type is well know and can be used inside fly() method.
// template<class Entity>
// void fly(Entity& entity)
// {
// Alternative 3: Abbreviated Function Template (Modern C++20)
// The Entity type is deduced at compile-time.
// Any class that provides 'tank_get_fuel' is accepted at compilation time.
// The Entity type isn't know inside fly() method.
void fly(auto& entity)
{
if(entity.tank_get_fuel(1)) std::cout << "\tFlying at " << altitude_ << " ft\n";
else                         std::cout << "\tNo fuel to Fly\n";
}
void print() const 
{
std::cout << "  - Fly capability: Alt " << altitude_ << " / Speed " << speed_ << "\n";
}
};
//-------------------------------------------------------------- Generic Entity:
//----------------------------------------------- Basic_Entity:
class Basic_Entity
{
private:
std::string name_;
public:
explicit Basic_Entity(std::string n) : name_{std::move(n)} { }
void print_name() const 
{
std::cout << "\n" << name_ << ":\n";
}
};
//----------------------------------------------------- Entity:
// Variadic Template Entity inheriting from all Mixins
template<class ... Mixins>
class Entity : public Basic_Entity, public Mixins...
{
public:
explicit Entity(std::string name) : Basic_Entity{std::move(name)}, Mixins{}... { }
// Advanced Visitor: Uses Fold Expressions to apply a function to all bases
template<typename Visitor>
void visitFeatures(Visitor visitor)
{
this->print_name();
// static_cast unpacks the 'this' pointer into each specific base class
visitor(static_cast<Mixins&>(*this) ...);
}
};
//--------------------------------------------------------------- Mixin Visitor:
struct PrintVisitor
{
template<class ... Mixins>
void operator()(Mixins& ... mixins) const
{
// C++17 Fold Expression: call print() on every mixin in the pack
(mixins.print(), ...);
}
};
//----------------------------------------------------------- Specific Entities:
//----------------------------------------------------- Dragon:
// Extension for Dragon to use fly() without arguments:
// using Dragon = public Entity<Fly, Tank, Laser>
class Dragon : public Entity<Fly, Tank, Laser>
{
public:
using Entity::Entity;
// Get Entity constructor
void fly() 
{
Fly::fly(*this);
}
};
Dragon createDragon(std::string name, int fuel, int intensity)
{
Dragon dragon{std::move(name)};
dragon.tank_load_fuel(fuel);
dragon.laser_set_intensity(intensity);
return dragon;
}
//--------------------------------------------------- Elephant:
using Elephant = Entity<Walk, Gun>;
Elephant createElephant(std::string name, int speed, int bullets)
{
Elephant elephant{std::move(name)};
elephant.walk_set_speed(speed);
elephant.gun_set_bullets(bullets);
return elephant;
}
//--------------------------------------------------- Airplane:
// Extension for Airplane to use fly() without arguments:
// using Airplane = Entity<Fly, Tank, Laser, Gun>
class Airplane : public Entity<Fly, Tank, Laser, Gun>
{
public:
using Entity::Entity;
// Get Entity constructor
void fly() 
{
Fly::fly(*this);
}
};
Airplane createAirplane(std::string name, int fuel, int intensity, int bullets)
{
Airplane airplane{std::move(name)};
airplane.fly_set_altitude_speed(5000, 600);
airplane.tank_load_fuel(fuel);
airplane.laser_set_intensity(intensity);
airplane.gun_set_bullets(bullets);
return airplane;
}
//------------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MIXIN PATTERN (ADVANCED VISITOR & DEPENDENCIES) ===\n";
//-------------------------------------------------- Dragon:
Dragon dragon = createDragon("Ancient Dragon", 2, 4);
dragon.visitFeatures(PrintVisitor{});
std::cout << "  + Testing dragon:" << std::endl;
dragon.fly();
dragon.laser_fire();
dragon.fly();
dragon.fly();
// Should fail (no fuel)
std::cout << "\tLoading fuel 1\n";
dragon.tank_load_fuel(1);
dragon.fly();
// Now it should work
dragon.laser_fire();
//------------------------------------------------ Elephant:
Elephant elephant = createElephant("War Elephant", 3, 2);
elephant.visitFeatures(PrintVisitor{});
std::cout << "  + Testing elephant:" << std::endl;
elephant.walk();
elephant.gun_fire();
elephant.gun_fire();
elephant.gun_fire();
//------------------------------------------------ Airplane:
Airplane airplane = createAirplane("Combat Jet", 2, 5, 3);
airplane.visitFeatures(PrintVisitor{});
std::cout << "  + Testing ariplane:" << std::endl;
airplane.fly();
airplane.laser_fire();
airplane.gun_fire();
airplane.gun_fire();
airplane.fly();
airplane.fly();
// Should fail
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//========================================================================== END
/**
* ============================================================================
* File: BlurProcessor.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* Concrete implementation of the Processor interface. Demonstrates how 
* easy it is to add new features to the system by simply including a new 
* file and registering it.
* ============================================================================
*/
#include "Register.h"
class BlurProcessor : public Processor
{
public:
Image process(const Image& image) const override
{
return image + " blured";
}
};
// Auto-register using the helper
static Register<BlurProcessor> reg("Blur");
//================================================================================ END
/**
* ============================================================================
* File: GrayscaleProcessor.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* Concrete implementation of the Processor interface. This component 
* automatically registers itself into the Registry upon initialization.
* ============================================================================
*/
#include "Register.h"
class GrayscaleProcessor : public Processor
{
public:
Image process(const Image& image) const override
{
return image + " grayscaled";
}
};
// Auto-register using the helper
static Register<GrayscaleProcessor> reg("Grayscale");
//================================================================================ END
/**
* ============================================================================
* File: Register.cpp (Registry Pattern)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Registry pattern, providing a centralized 
* lookup for service creation. The client requests processors by name 
* without knowing their concrete implementation.
* 
* --- AUTO-REGISTRATION:
* Each concrete class registers itself into the 'Registry' using the 
* 'Register<T>' helper during static initialization.
* 
* --- VERIFICATION:
* We test the registry by requesting the same processor twice. Depending on 
* the compilation flag (REGISTRY_SINGLETON or REGISTRY_PROTOTYPE), you can 
* verify if the instance returned is shared or a fresh copy.
* ============================================================================
*/
#include "Register.h"
#include <iostream>
#include <string>
int main()
{
std::cout << "=== REGISTRY PATTERN SIMULATION ===\n" << std::endl;
Image image = "I am an image";
std::cout << "Original: " << image << "\n\n";
try
{
// The client asks the Registry for processors by name
auto p1 = Registry::create("Grayscale");
auto p2 = Registry::create("Blur");
// Process the image
image = p1->process(image);
std::cout << "Result: " << image << std::endl;
image = p2->process(image);
std::cout << "Result: " << image << std::endl;
// Verification: Check if repeated request returns the same instance
std::cout << "\n--- Verifying Instance Policy ---\n";
auto p3 = Registry::create("Grayscale");
std::cout << " Pointer 1 (p1): " << p1.get() << "\n";
std::cout << " Pointer 2 (p3): " << p3.get() << "\n";
if(p1 == p3) std::cout << " Policy: Singleton (Instance shared)";
else         std::cout << " Policy: Prototype (New instance created)";
std::cout << " - Defined in Register.h\n";
// Testing non-existent processor
std::cout << "\nAttempting to get an unknown processor:\n";
Registry::create("Sepia");
}
catch (const std::exception& e)
{
std::cerr << " [Error] " << e.what() << "\n";
}
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Module.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This is the concrete Plugin implementation. It gets compiled into a 
* Shared Object (.so). It exports two flat C functions (build_module 
* and destroy_module) to bypass C++ name mangling, allowing the Host 
* to instantiate and destroy the object safely.
* ============================================================================
*/
//------------------------ Module implementation
#include "IModule.h"
#include <iostream>
#include <string>
class Module : public IModule
{
private:
std::string name_;
int factor_;
public:
// Constructor now accepts configuration from the Host
explicit Module(const char* name, const int factor) : name_{name}, factor_{factor}
{
std::cout << "    -> [Plugin] Module '" << name_ << "' initialized.\n";
std::cout << "    -> [Plugin] with factor = " << factor_ << "\n";
}
~Module() override
{
std::cout << "    -> [Plugin] Module '" << name_ << "' destroyed.\n";
}
int processData(int data) override
{
std::cout << "    -> [Plugin] Processing data: " << data << "\n";
std::cout << "    -> [Plugin] Returning calculated value: " << data*factor_ << "\n";
return data*factor_;
}
};
// The following "C" functions must be into the library:
extern "C"
{
IModule* build_module(const char* name, const int factor)
{
std::cout << "    -> [Plugin] build_module called.\n";
return new Module(name, factor);
}
void destroy_module(IModule* module_ptr)
{
if(module_ptr)
{
std::cout << "    -> [Plugin] destroy_module called.\n";
delete module_ptr;
}
}
}
//========================================================================= END
/**
* ============================================================================
* File: ModuleLoader.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This is the Host application. It dynamically loads 'libModule.so' 
* at runtime using the POSIX <dlfcn.h> API.
* 
* --- MODERN C++ IMPROVEMENTS:
* 1. RAII Library Handle: A small wrapper ensures 'dlclose' is always 
*    called, preventing resource leaks even if exceptions occur.
* 2. Smart Pointers with Custom Deleter: 'std::unique_ptr' is used 
*    with the plugin's 'destroy_module' function to guarantee safe 
*    deallocation across the ABI boundary.
* ============================================================================
*/
// Load a module (an object from a .so file) and call a method from this module.
#include "IModule.h"
#include <dlfcn.h>
// This is a C library
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
// RAII wrapper for the dynamic library handle
class DynamicLibrary
{
private:
void* libraryHandle_{nullptr};
public:
explicit DynamicLibrary(const char* filename)
{
std::cout << " [Host] Loading shared library: " << filename << "...\n";
// RTLD_LAZY: Relocations are performed at an implementation-defined time.
libraryHandle_ = dlopen(filename, RTLD_LAZY);
if(!libraryHandle_) throw std::runtime_error(dlerror());
}
~DynamicLibrary()
{
if(libraryHandle_)
{
std::cout << " [Host] Unloading shared library...\n";
dlclose(libraryHandle_);
}
}
void* getSymbol(const char* symbolName) const
{
void* symbol = dlsym(libraryHandle_, symbolName);
if(!symbol) throw std::runtime_error(dlerror());
return symbol;
}
};
int main(int ac, char** av)
{
std::cout << "=== DYNAMIC MODULE LOADER ===\n\n";
//---------------------------------------------------------------- Verify input:
if(ac < 2)
{
std::cerr << "usage: ModuleLoader ./libModule.so\n";
return 1;
}
dlerror();
// Clear all previous errors
try
{
//-------------------------------------------------------------- Open a library:
// Utilizing RAII to ensure the library is closed automatically
DynamicLibrary lib(av[1]);
//---------------------------------------------------------------- Define types:
// Module_constructor is a pointer to a function with parameters
// (const char*, const int) and returnnig a pointer to IModule:
using Module_constructor = IModule* (*)(const char*, const int);
// Module_destructor is a pointer to a function with parameter
// (IModule*) and returning void:
using Module_destructor = void (*)(IModule*);
//---------------------------------------- Get build and destroy module symbols:
std::cout << " [Host] Resolving symbols (build_module, destroy_module)...\n";
auto build_module   = reinterpret_cast<Module_constructor>(lib.getSymbol("build_module"));
auto destroy_module = reinterpret_cast<Module_destructor> (lib.getSymbol("destroy_module"));
//------------------------------------------------ Create a new module instance:
std::cout << " [Host] Requesting new module instance...\n";
IModule* raw_module = build_module("MathProcessor_v1", 2);
if(!raw_module)
{
std::cerr << "build(): returned NULL instead of a new module\n";
return 1;
}
// Bind the raw pointer to a unique_ptr with a custom deleter: 'destroy_module'.
// This guarantees 'destroy_module' is called when the pointer goes out of scope.
std::unique_ptr<IModule, Module_destructor> module(raw_module, destroy_module);
//------------------------------------------------------- Call Module functions:
std::cout << "\n [Host] Executing module operation with data: 42...\n";
int answer = module->processData(42);
std::cout << " [Host] Module returned value: " << answer << "\n";
std::cout << "\n [Host] Destroying module instance...\n";
//--------------------------------------------------------------- Delete Module:
// The unique_ptr will automatically call destroy_module(module.get()) here 
// when it goes out of scope, right before the DynamicLibrary is destroyed.
}
catch(const std::exception& e)
{
std::cerr << "[Error] " << e.what() << "\n";
return 1;
}
std::cout << "\n=== EXECUTION COMPLETED ===\n";
}
//======================================================================================== END
/**
* ============================================================================
* File: ChainOfResponsability.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Chain of Responsibility pattern. Each handler 
* in the chain decides whether to process a request or pass it to the next 
* successor.
* 
* --- CHAIN LINKING:
* We construct the chain using std::unique_ptr, where each handler 'owns' 
* the next one. When the head of the chain is destroyed, the entire chain 
* is cleaned up automatically.
* 
* --- BUILDER APPROACH:
* The Chain class acts as a builder to construct the chain fluently. 
* Handlers are linked dynamically, allowing the chain to be assembled 
* in stages or at runtime while maintaining the correct priority order.
* 
* --- CHAIN LOGIC:
* 1. Each 'Handler' knows about the 'next' handler.
* 2. If a handler cannot process the request, it delegates to 'next'.
* 3. The 'Chain' builder simplifies the construction process, maintaining 
*    the order of priority (first added, first handled).
* ============================================================================
*/
#include <iostream>
#include <memory>
#include <string>
#include <utility>
//--------------------------------------------------------- Base Handler:
class IHandler
{
protected:
std::unique_ptr<IHandler> next_;
public:
virtual ~IHandler() = default;
void setNext(std::unique_ptr<IHandler> next)
{
next_ = std::move(next);
}
virtual void handle(int request)
{
if(next_) next_->handle(request);
else      std::cout << " [System] Request " << request 
<< " reached the end of the chain unhandled.\n";
}
};
//-------------------------------------------------------- Concrete Handlers:
class Handler_1 : public IHandler
{
private:
int id_;
std::string name_;
public:
Handler_1(int id, std::string name) : id_{id}, name_{std::move(name)} { }
void handle(int request) override
{
if(request == id_)
std::cout << " [Handler_1] " << name_ << " handled request: " << request << "\n";
else
{
std::cout << " [Handler_1] " << name_ << " passing request " << request << " forward.\n";
IHandler::handle(request);
}
}
};
class Handler_2 : public IHandler
{
private:
int id_;
std::string name_;
public:
Handler_2(int id, std::string name) : id_{id}, name_{std::move(name)} { }
void handle(int request) override
{
if(request == id_)
std::cout << " [Handler_2] " << name_ << " handled request: " << request << "\n";
else
{
std::cout << " [Handler_2] " << name_ << " passing request " << request << " forward.\n";
IHandler::handle(request);
}
}
};
//--------------------------------------------------------- Chain Builder:
class Chain
{
private:
std::unique_ptr<IHandler> head_;
IHandler* tail_{nullptr};
public:
Chain& add(std::unique_ptr<IHandler> handler)
{
IHandler* current = handler.get();
if(!head_) head_ = std::move(handler);
else       tail_->setNext(std::move(handler));
tail_ = current;
return *this;
}
void execute(int request) const
{
if(head_) head_->handle(request);
else      std::cout << "Error: There are not chain to handle " << request << std::endl;
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== CHAIN OF RESPONSIBILITY (CHAINBUILDER) ===\n" << std::endl;
Chain chain;
// 1. First stage of construction
chain.add(std::make_unique<Handler_1>(3, "Handler-3"))
.add(std::make_unique<Handler_2>(5, "Handler-5"));
// Some code...
// 2. Second stage of construction
chain.add(std::make_unique<Handler_2>(8, "Handler-8"))
.add(std::make_unique<Handler_1>(11, "Handler-11"));
// Execute tests
int requests[] = {3, 5, 4, 8, 11};
for(int r : requests)
{
std::cout << "Testing request " << r << ":\n";
chain.execute(r);
std::cout << std::endl;
}
std::cout << "=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Command.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation uses a generic 'CommandQueue' that manages a queue of 
* 'Command' objects. The 'Receivers' (Cow, Dog, Car) remain decoupled from 
* the 'CommandQueue'.
* 
* --- MEMORY MANAGEMENT:
* The 'CommandQueue' owns the commands via 'std::unique_ptr', ensuring 
* automatic cleanup of all issued requests.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <utility>
//--------------------------------------------------------- Receivers:
class Cow
{
public:
void moo() const { std::cout << " [Receiver] Cow says moo!\n"; }
};
class Dog
{
private:
std::string name_;
public:
explicit Dog(std::string name) : name_{std::move(name)} { }
void bark() const { std::cout << " [Receiver] " << name_ << " barks!\n"; }
};
class Car
{
private:
bool engineOn_{false};
public:
void turnOn()  const { std::cout << " [Receiver] Car engine ON.\n"; }
void turnOff() const { std::cout << " [Receiver] Car engine OFF.\n"; }
void rev()     const { std::cout << " [Receiver] Car: Vroom, vroom!\n"; }
};
//--------------------------------------------------------- Command Interface:
class ICommand
{
public:
virtual ~ICommand() = default;
virtual void execute() const = 0;
};
//--------------------------------------------------------- Concrete Commands:
class CowCommand : public ICommand
{
private:
Cow& receiver_;
public:
explicit CowCommand(Cow& receiver) : receiver_{receiver} { }
void execute() const override { receiver_.moo(); }
};
class DogCommand : public ICommand
{
private:
Dog& receiver_;
public:
explicit DogCommand(Dog& receiver) : receiver_{receiver} { }
void execute() const override { receiver_.bark(); }
};
class CarCommand : public ICommand
{
private:
Car& receiver_;
public:
explicit CarCommand(Car& receiver) : receiver_{receiver} { }
void execute() const override
{
receiver_.turnOn();
receiver_.rev();
receiver_.turnOff();
}
};
//---------------------------------------------------- CommandQueue:
class CommandQueue
{
private:
std::vector<std::unique_ptr<ICommand>> queue_;
public:
void addCommand(std::unique_ptr<ICommand> cmd) { queue_.push_back(std::move(cmd)); }
void runAll() const
{
for(const auto& command : queue_) command->execute();
}
};
//--------------------------------------------------------- Main:
int main()
{
std::cout << "=== COMMAND PATTERN SIMULATION ===\n" << std::endl;
Cow cow;
Dog dog{"Marshall"};
Car car;
CommandQueue commandQueue;
commandQueue.addCommand(std::make_unique<CowCommand>(cow));
commandQueue.addCommand(std::make_unique<DogCommand>(dog));
commandQueue.addCommand(std::make_unique<CarCommand>(car));
commandQueue.addCommand(std::make_unique<DogCommand>(dog));
std::cout << "Executing command queue:\n";
commandQueue.runAll();
std::cout << "\nExecuting command queue (again):\n";
commandQueue.runAll();
std::cout << "\n--- Executing command outside the queue ---\n";
std::make_unique<CarCommand>(car)->execute();
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Command.cpp (Modern Variant Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates a non-intrusive Command pattern using std::variant.
* Commands are represented as simple data structures, and their execution
* is dispatched via std::visit.
* 
* --- ADVANTAGES:
* 1. No inheritance required for commands.
* 2. Commands are stored by value in the queue (no pointers/new/delete).
* 3. The execution logic is centralized in the 'CommandExecutor' visitor.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <string>
#include <variant>
//--------------------------------------------------------- Receivers:
class Cow
{
public:
void moo() const { std::cout << " [Receiver] Cow says moo!\n"; }
};
class Dog
{
private:
std::string name_;
public:
explicit Dog(std::string name) : name_{std::move(name)} { }
void bark() const { std::cout << " [Receiver] " << name_ << " barks!\n"; }
};
class Car
{
public:
void turnOn()  const { std::cout << " [Receiver] Car engine ON.\n"; }
void turnOff() const { std::cout << " [Receiver] Car engine OFF.\n"; }
void rev()     const { std::cout << " [Receiver] Car: Vroom, vroom!\n"; }
};
//--------------------------------------------------------- Command Data:
// Commands are now just simple empty structs (Data markers)
struct CowCommand { };
struct DogCommand { };
struct CarCommand { };
using Command = std::variant<CowCommand, DogCommand, CarCommand>;
//--------------------------------------------------------- Command Executor:
// This is the "Visitor" that knows how to execute
// each command data type and owns the receivers.
class CommandExecutor
{
private:
Cow cow;
Dog dog{"Marshall"};
Car car;
public:
void operator()(const CowCommand&) const { cow.moo(); }
void operator()(const DogCommand&) const { dog.bark(); }
void operator()(const CarCommand&) const
{
car.turnOn();
car.rev();
car.turnOff();
}
};
//---------------------------------------------------- CommandQueue:
class CommandQueue
{
private:
std::vector<Command> queue_;
CommandExecutor& commandExecutor_;
public:
CommandQueue(CommandExecutor& commandExecutor) : commandExecutor_{commandExecutor} {}
void addCommand(Command cmd) { queue_.push_back(std::move(cmd)); }
void runAll() const
{
for(const auto& command : queue_) std::visit(commandExecutor_, command);
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MODERN VARIANT COMMAND PATTERN SIMULATION ===\n" << std::endl;
CommandExecutor commandExecutor;
CommandQueue commandQueue{commandExecutor};
commandQueue.addCommand(CowCommand{});
commandQueue.addCommand(DogCommand{});
commandQueue.addCommand(CarCommand{});
commandQueue.addCommand(DogCommand{});
std::cout << "Executing command queue:\n";
commandQueue.runAll();
std::cout << "\nExecuting command queue (again):\n";
commandQueue.runAll();
std::cout << "\n--- Executing command outside the queue ---\n";
std::visit(commandExecutor, Command{CarCommand{}});
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: calc.cpp (Interpreter: GoF AST Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation follows the classic Gang of Four Interpreter pattern.
* 1. Parser (Recursive Descent): Reads the string and builds an Abstract 
*    Syntax Tree (AST) of polymorphic objects in memory.
* 2. Evaluation: We call evaluate() on the root node, which recursively 
*    calls evaluate() on its children to compute the final result.
* 
* Usage: ./calc [-p] "<math_expression>"
* The '-p' flag prints the object tree (AST) before evaluating.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <cctype>
//=============================================================================
// BLOCK 1: LEXER (Scanner)
//=============================================================================
enum class TokenType 
{ 
NUMBER, FUNCTION, PLUS, MINUS, MUL, DIV, POW, LPAREN, RPAREN, END_OF_FILE 
};
struct Token 
{
TokenType   type;
double      value {0.0};
std::string name  {""};
};
class Lexer
{
private:
std::string input_;
size_t      pos_{0};
Token       currentToken_;
void advanceChar() { pos_++; }
char peek() const { return (pos_ < input_.size()) ? input_[pos_] : '\0'; }
public:
explicit Lexer(std::string input) : input_{std::move(input)} 
{
nextToken();
// Load first token
}
Token getToken() const { return currentToken_; }
void nextToken()
{
while(isspace(peek())) advanceChar();
if(pos_ >= input_.size())
{
currentToken_ = {TokenType::END_OF_FILE};
return;
}
char c = peek();
if(isdigit(c) || c == '.')
{
size_t length;
double val = std::stod(input_.substr(pos_), &length);
pos_ += length;
currentToken_ = {TokenType::NUMBER, val, ""};
}
else if(isalpha(c))
{
std::string funcName = "";
while(isalpha(peek()))
{
funcName += peek();
advanceChar();
}
currentToken_ = {TokenType::FUNCTION, 0.0, funcName};
}
else
{
switch(c)
{
case '+': currentToken_ = {TokenType::PLUS};   break;
case '-': currentToken_ = {TokenType::MINUS};  break;
case '*': currentToken_ = {TokenType::MUL};    break;
case '/': currentToken_ = {TokenType::DIV};    break;
case '^': currentToken_ = {TokenType::POW};    break;
case '(': currentToken_ = {TokenType::LPAREN}; break;
case ')': currentToken_ = {TokenType::RPAREN}; break;
default:  throw std::runtime_error(std::string("Syntax error: Unknown character: ") + c);
}
advanceChar();
}
// else
}
// nextToken
};
// class Lexer
//=============================================================================
// BLOCK 2: ABSTRACT SYNTAX TREE (GoF Interpreter Nodes)
//=============================================================================
class IExpression
{
public:
virtual ~IExpression() = default;
virtual double evaluate()          const = 0;
virtual void print(int indent = 0) const = 0;
protected:
void printIndent(int indent) const
{
std::cout << "    ";
for(int i = 0; i < indent; ++i) std::cout << "   ";
}
};
using ExprPtr = std::unique_ptr<IExpression>;
// --- Terminal IExpression ---
class NumberNode : public IExpression
{
private:
double value_;
public:
explicit NumberNode(double value) : value_{value} {}
double evaluate() const override { return value_; }
void print(int indent) const override 
{ 
printIndent(indent); 
std::cout << "Number: " << value_ << "\n"; 
}
};
//-------------------------------------------------------------------------------- BinaryExpression :
class BinaryExpression : public IExpression
{
protected:
ExprPtr left_;
ExprPtr right_;
public:
BinaryExpression(ExprPtr left, ExprPtr right) 
: left_{std::move(left)}, right_{std::move(right)} {}
};
class AddNode : public BinaryExpression
{
public:
using BinaryExpression::BinaryExpression;
double evaluate() const override { return left_->evaluate() + right_->evaluate(); }
void print(int indent) const override
{
printIndent(indent); std::cout << "Add\n";
left_->print(indent + 1);
right_->print(indent + 1);
}
};
class SubNode : public BinaryExpression
{
public:
using BinaryExpression::BinaryExpression;
double evaluate() const override { return left_->evaluate() - right_->evaluate(); }
void print(int indent) const override
{
printIndent(indent); std::cout << "Subtract\n";
left_->print(indent + 1);
right_->print(indent + 1);
}
};
class MulNode : public BinaryExpression
{
public:
using BinaryExpression::BinaryExpression;
double evaluate() const override { return left_->evaluate() * right_->evaluate(); }
void print(int indent) const override
{
printIndent(indent); std::cout << "Multiply\n";
left_->print(indent + 1);
right_->print(indent + 1);
}
};
class DivNode : public BinaryExpression
{
public:
using BinaryExpression::BinaryExpression;
double evaluate() const override 
{ 
double r = right_->evaluate();
if(r == 0.0) throw std::runtime_error("Error: Division by zero");
return left_->evaluate() / r; 
}
void print(int indent) const override
{
printIndent(indent); std::cout << "Divide\n";
left_->print(indent + 1);
right_->print(indent + 1);
}
};
class PowNode : public BinaryExpression
{
public:
using BinaryExpression::BinaryExpression;
double evaluate() const override { return std::pow(left_->evaluate(), right_->evaluate()); }
void print(int indent) const override
{
printIndent(indent); std::cout << "Power\n";
left_->print(indent + 1);
right_->print(indent + 1);
}
};
class UnaryMinusNode : public IExpression
{
private:
ExprPtr expr_;
public:
explicit UnaryMinusNode(ExprPtr expr) : expr_{std::move(expr)} {}
double evaluate() const override { return -expr_->evaluate(); }
void print(int indent) const override
{
printIndent(indent); std::cout << "UnaryMinus\n";
expr_->print(indent + 1);
}
};
class MathFunctionNode : public IExpression
{
private:
std::string func_;
ExprPtr expr_;
public:
MathFunctionNode(std::string func, ExprPtr expr) 
: func_{std::move(func)}, expr_{std::move(expr)} {}
double evaluate() const override
{
double val = expr_->evaluate();
if(func_ == "sin")  return std::sin(val);
if(func_ == "cos")  return std::cos(val);
if(func_ == "tan")  return std::tan(val);
if(func_ == "asin") return std::asin(val);
if(func_ == "acos") return std::acos(val);
if(func_ == "atan") return std::atan(val);
if(func_ == "exp")  return std::exp(val);
if(func_ == "sqrt")
{
if(val < 0) throw std::runtime_error("Error: sqrt of negative number");
return std::sqrt(val);
}
if(func_ == "ln")
{
if(val <= 0) throw std::runtime_error("Error: ln of non-positive number");
return std::log(val);
}
if(func_ == "log")
{
if(val <= 0) throw std::runtime_error("Error: log of non-positive number");
return std::log10(val);
}
throw std::runtime_error("Error: Unknown function: " + func_);
}
void print(int indent) const override
{
printIndent(indent); std::cout << "Function [" << func_ << "]\n";
expr_->print(indent + 1);
}
};
//=============================================================================
// BLOCK 3: PARSER (Builds the AST)
//=============================================================================
class Parser
{
private:
Lexer lexer_;
void match(TokenType expected)
{
if(lexer_.getToken().type == expected) lexer_.nextToken();
else throw std::runtime_error("Syntax error: Unexpected token");
}
ExprPtr parsePrimary()
{
Token t = lexer_.getToken();
if(t.type == TokenType::NUMBER)
{
match(TokenType::NUMBER);
return std::make_unique<NumberNode>(t.value);
}
if(t.type == TokenType::LPAREN)
{
match(TokenType::LPAREN);
ExprPtr expr = parseExpression();
match(TokenType::RPAREN);
return expr;
}
if(t.type == TokenType::FUNCTION)
{
std::string func = t.name;
match(TokenType::FUNCTION);
match(TokenType::LPAREN);
ExprPtr expr = parseExpression();
match(TokenType::RPAREN);
return std::make_unique<MathFunctionNode>(func, std::move(expr));
}
throw std::runtime_error("Syntax error in primary expression");
}
ExprPtr parseSigned()
{
Token t = lexer_.getToken();
if(t.type == TokenType::PLUS)
{
match(TokenType::PLUS);
return parsePrimary();
}
if(t.type == TokenType::MINUS)
{
match(TokenType::MINUS);
return std::make_unique<UnaryMinusNode>(parsePrimary());
}
return parsePrimary();
}
ExprPtr parseExponent()
{
ExprPtr left = parseSigned();
while(lexer_.getToken().type == TokenType::POW)
{
match(TokenType::POW);
ExprPtr right = parseSigned();
left = std::make_unique<PowNode>(std::move(left), std::move(right));
}
return left;
}
ExprPtr parseMultiplicative()
{
ExprPtr left = parseExponent();
while(lexer_.getToken().type == TokenType::MUL || 
lexer_.getToken().type == TokenType::DIV)
{
TokenType op = lexer_.getToken().type;
match(op);
ExprPtr right = parseExponent();
if(op == TokenType::MUL) left = std::make_unique<MulNode>(std::move(left), std::move(right));
else                     left = std::make_unique<DivNode>(std::move(left), std::move(right));
}
return left;
}
ExprPtr parseAdditive()
{
ExprPtr left = parseMultiplicative();
while(lexer_.getToken().type == TokenType::PLUS || 
lexer_.getToken().type == TokenType::MINUS)
{
TokenType op = lexer_.getToken().type;
match(op);
ExprPtr right = parseMultiplicative();
if(op == TokenType::PLUS) left = std::make_unique<AddNode>(std::move(left), std::move(right));
else                      left = std::make_unique<SubNode>(std::move(left), std::move(right));
}
return left;
}
ExprPtr parseExpression()
{
return parseAdditive();
}
public:
explicit Parser(const std::string& source) : lexer_(source) { }
ExprPtr parse()
{
ExprPtr tree = parseExpression();
if(lexer_.getToken().type != TokenType::END_OF_FILE)
throw std::runtime_error("Syntax error: Unexpected tokens at end");
return tree;
}
};
//=============================================================================
// BLOCK 4: COMMAND LINE INTERFACE (Main)
//=============================================================================
int main(int argc, char* argv[])
{
std::cout << "=== INTERPRETER PATTERN (GoF AST VERSION) ===\n" << std::endl;
if(argc < 2 || argc > 3)
{
std::cerr << "Usage: ./calc [-p] \"<math_expression>\"\n"
<< "Note: Trigonometric functions are in radians\n";
return 1;
}
bool printInternals = false;
std::string sourceCode;
if(argc == 3)
{
std::string flag = argv[1];
if(flag == "-p") printInternals = true;
else
{
std::cerr << "Error: Unknown flag '" << flag << "'. Only '-p' is supported.\n";
return 1;
}
sourceCode = argv[2];
}
else sourceCode = argv[1];
try
{
// 1. Parse source code into an Abstract Syntax Tree
Parser parser(sourceCode);
ExprPtr astRoot = parser.parse();
// 2. Optionally print the AST
if(printInternals)
{
std::cout << "--- Abstract Syntax Tree ---\n";
astRoot->print();
std::cout << "----------------------------\n\n";
}
// 3. Evaluate the AST recursively and print the result
double result = astRoot->evaluate();
std::cout << "Result: " << result << "\n";
}
catch(const std::exception& e)
{
std::cerr << e.what() << "\n";
return 1;
}
}
//================================================================================ END
/**
* ============================================================================
* File: calc.cpp (Interpreter: Stack Machine Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This is an industrial-style interpreter. It does not use the GoF AST.
* Instead, it works in two phases:
* 1. Compiler (Recursive Descent): Parses the math string and generates 
*    a linear array of instructions (Bytecode).
* 2. Virtual Machine: Evaluates the bytecode using a Stack.
* 
* Usage: ./calc [-p] "<math_expression>"
* The '-p' flag prints the generated bytecode before evaluating.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cctype>
//=============================================================================
// BLOCK 1: BYTECODE & TOKENS DEFINITIONS
//=============================================================================
enum class OpCode 
{ 
PUSH, ADD, SUB, MUL, DIV, POW, CHSGN, 
SIN, COS, TAN, ASIN, ACOS, ATAN, SQRT, LN, LOG, EXP 
};
struct Instruction 
{
OpCode code;
double value{0.0};
// Only used if code is PUSH
};
enum class TokenType 
{ 
NUMBER, FUNCTION, PLUS, MINUS, MUL, DIV, POW, LPAREN, RPAREN, END_OF_FILE 
};
struct Token 
{
TokenType type;
double value{0.0};
std::string name{""};
};
//=============================================================================
// BLOCK 2: LEXER (Scanner)
//=============================================================================
class Lexer
{
private:
std::string input_;
size_t pos_{0};
Token currentToken_;
void advanceChar() { pos_++; }
char peek() const { return (pos_ < input_.size()) ? input_[pos_] : '\0'; }
public:
explicit Lexer(std::string input) : input_{std::move(input)} 
{
nextToken();
// Load first token
}
Token getToken() const { return currentToken_; }
void nextToken()
{
while(isspace(peek())) advanceChar();
if(pos_ >= input_.size())
{
currentToken_ = {TokenType::END_OF_FILE};
return;
}
char c = peek();
if(isdigit(c) || c == '.')
{
size_t length;
double val = std::stod(input_.substr(pos_), &length);
pos_ += length;
currentToken_ = {TokenType::NUMBER, val, ""};
}
else if(isalpha(c))
{
std::string funcName = "";
while(isalpha(peek()))
{
funcName += peek();
advanceChar();
}
currentToken_ = {TokenType::FUNCTION, 0.0, funcName};
}
else
{
switch(c)
{
case '+': currentToken_ = {TokenType::PLUS}; break;
case '-': currentToken_ = {TokenType::MINUS}; break;
case '*': currentToken_ = {TokenType::MUL}; break;
case '/': currentToken_ = {TokenType::DIV}; break;
case '^': currentToken_ = {TokenType::POW}; break;
case '(': currentToken_ = {TokenType::LPAREN}; break;
case ')': currentToken_ = {TokenType::RPAREN}; break;
default:  throw std::runtime_error(std::string("Syntax error: Unknown character: ") + c);
}
advanceChar();
}
}
};
//=============================================================================
// BLOCK 3: COMPILER (Recursive Descent Parser)
//=============================================================================
class Compiler
{
private:
Lexer lexer_;
std::vector<Instruction> program_;
void match(TokenType expected)
{
if(lexer_.getToken().type == expected) lexer_.nextToken();
else throw std::runtime_error("Syntax error: Unexpected token");
}
void emit(OpCode code, double value = 0.0)
{
program_.push_back({code, value});
}
// --- Grammar Rules ---
void parsePrimary()
{
Token t = lexer_.getToken();
if(t.type == TokenType::NUMBER)
{
emit(OpCode::PUSH, t.value);
match(TokenType::NUMBER);
}
else if(t.type == TokenType::LPAREN)
{
match(TokenType::LPAREN);
parseExpression();
match(TokenType::RPAREN);
}
else if(t.type == TokenType::FUNCTION)
{
std::string func = t.name;
match(TokenType::FUNCTION);
match(TokenType::LPAREN);
parseExpression();
match(TokenType::RPAREN);
if(func == "sin")       emit(OpCode::SIN);
else if(func == "cos")  emit(OpCode::COS);
else if(func == "tan")  emit(OpCode::TAN);
else if(func == "asin") emit(OpCode::ASIN);
else if(func == "acos") emit(OpCode::ACOS);
else if(func == "atan") emit(OpCode::ATAN);
else if(func == "sqrt") emit(OpCode::SQRT);
else if(func == "ln")   emit(OpCode::LN);
else if(func == "log")  emit(OpCode::LOG);
else if(func == "exp")  emit(OpCode::EXP);
else throw std::runtime_error("Error: Unknown function: " + func);
}
else throw std::runtime_error("Syntax error in primary expression");
}
void parseSigned()
{
Token t = lexer_.getToken();
if(t.type == TokenType::PLUS)
{
match(TokenType::PLUS);
parsePrimary();
}
else if(t.type == TokenType::MINUS)
{
match(TokenType::MINUS);
parsePrimary();
emit(OpCode::CHSGN);
}
else parsePrimary();
}
void parseExponent()
{
parseSigned();
while(lexer_.getToken().type == TokenType::POW)
{
match(TokenType::POW);
parseSigned();
emit(OpCode::POW);
}
}
void parseMultiplicative()
{
parseExponent();
while(lexer_.getToken().type == TokenType::MUL || 
lexer_.getToken().type == TokenType::DIV)
{
TokenType op = lexer_.getToken().type;
match(op);
parseExponent();
if(op == TokenType::MUL) emit(OpCode::MUL);
else emit(OpCode::DIV);
}
}
void parseAdditive()
{
parseMultiplicative();
while(lexer_.getToken().type == TokenType::PLUS || 
lexer_.getToken().type == TokenType::MINUS)
{
TokenType op = lexer_.getToken().type;
match(op);
parseMultiplicative();
if(op == TokenType::PLUS) emit(OpCode::ADD);
else emit(OpCode::SUB);
}
}
void parseExpression()
{
parseAdditive();
}
public:
explicit Compiler(const std::string& source) : lexer_(source) { }
std::vector<Instruction> compile()
{
parseExpression();
if(lexer_.getToken().type != TokenType::END_OF_FILE)
throw std::runtime_error("Syntax error: Unexpected tokens at end");
return program_;
}
};
//=============================================================================
// BLOCK 4: VIRTUAL MACHINE (Evaluator)
//=============================================================================
class VirtualMachine
{
private:
std::string opcodeToString(OpCode code) const
{
switch(code)
{
case OpCode::PUSH:  return "PUSH";
case OpCode::ADD:   return "ADD";
case OpCode::SUB:   return "SUB";
case OpCode::MUL:   return "MUL";
case OpCode::DIV:   return "DIV";
case OpCode::POW:   return "POW";
case OpCode::CHSGN: return "CHSGN";
case OpCode::SIN:   return "SIN";
case OpCode::COS:   return "COS";
case OpCode::TAN:   return "TAN";
case OpCode::ASIN:  return "ASIN";
case OpCode::ACOS:  return "ACOS";
case OpCode::ATAN:  return "ATAN";
case OpCode::SQRT:  return "SQRT";
case OpCode::LN:    return "LN";
case OpCode::LOG:   return "LOG";
case OpCode::EXP:   return "EXP";
default:            return "UNKNOWN";
}
}
public:
void printProgram(const std::vector<Instruction>& program) const
{
std::cout << "--- Bytecode Program ---\n";
for(const auto& inst : program)
{
std::cout << "  " << opcodeToString(inst.code);
if(inst.code == OpCode::PUSH) std::cout << " " << inst.value;
std::cout << "\n";
}
std::cout << "------------------------\n\n";
}
double evaluate(const std::vector<Instruction>& program) const
{
std::vector<double> stack;
for(const auto& inst : program)
{
if(inst.code == OpCode::PUSH)
{
stack.push_back(inst.value);
continue;
}
// Handle Unary Operations
if(inst.code >= OpCode::CHSGN)
{
if(stack.empty()) throw std::runtime_error("Error: Stack underflow on unary op");
double val = stack.back();
stack.pop_back();
switch(inst.code)
{
case OpCode::CHSGN: stack.push_back(-val);           break;
case OpCode::SIN:   stack.push_back(std::sin(val));  break;
case OpCode::COS:   stack.push_back(std::cos(val));  break;
case OpCode::TAN:   stack.push_back(std::tan(val));  break;
case OpCode::ASIN:  stack.push_back(std::asin(val)); break;
case OpCode::ACOS:  stack.push_back(std::acos(val)); break;
case OpCode::ATAN:  stack.push_back(std::atan(val)); break;
case OpCode::EXP:   stack.push_back(std::exp(val));  break;
case OpCode::SQRT:  
if(val < 0) throw std::runtime_error("Error: sqrt of negative number");
stack.push_back(std::sqrt(val)); break;
case OpCode::LN:
if(val <= 0) throw std::runtime_error("Error: ln of non-positive number");
stack.push_back(std::log(val)); break;
case OpCode::LOG:
if(val <= 0) throw std::runtime_error("Error: log of non-positive number");
stack.push_back(std::log10(val)); break;
default: break;
}
continue;
}
// Handle Binary Operations
if(stack.size() < 2) throw std::runtime_error("Error: Stack underflow on binary op");
double right = stack.back(); stack.pop_back();
double left  = stack.back(); stack.pop_back();
switch(inst.code)
{
case OpCode::ADD: stack.push_back(left + right); break;
case OpCode::SUB: stack.push_back(left - right); break;
case OpCode::MUL: stack.push_back(left * right); break;
case OpCode::DIV: 
if(right == 0.0) throw std::runtime_error("Error: Division by zero");
stack.push_back(left / right); break;
case OpCode::POW: stack.push_back(std::pow(left, right)); break;
default: break;
}
}
if(stack.size() != 1) throw std::runtime_error("Error: Unbalanced evaluation stack");
return stack.back();
}
};
//=============================================================================
// BLOCK 5: COMMAND LINE INTERFACE (Main)
//=============================================================================
int main(int argc, char* argv[])
{
std::cout << "=== INTERPRETER PATTERN (STACK MACHINE) ===\n" << std::endl;
if(argc < 2 || argc > 3)
{
std::cerr << "Usage: calc [-p] \"<math_expression>\"\n"
<< "Note: Trigonometric functions are in radians\n";
return 1;
}
bool printInternals = false;
std::string sourceCode;
if(argc == 3)
{
std::string flag = argv[1];
if(flag == "-p")
{
printInternals = true;
sourceCode = argv[2];
}
else
{
std::cerr << "Error: Unknown flag '" << flag << "'. Only '-p' is supported.\n";
return 1;
}
}
else
{
sourceCode = argv[1];
}
try
{
// 1. Compile source code into bytecode
Compiler compiler(sourceCode);
std::vector<Instruction> program = compiler.compile();
VirtualMachine vm;
// 2. Optionally print the bytecode
if(printInternals) vm.printProgram(program);
// 3. Evaluate the bytecode and print the result
double result = vm.evaluate(program);
std::cout << "Result: " << result << "\n";
}
catch(const std::exception& e)
{
std::cerr << e.what() << "\n";
return 1;
}
}
//================================================================================ END
/**
* ============================================================================
* File: Iterator.cpp (Classic GoF Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the classic Object-Oriented Iterator pattern.
* We build a custom collection (a Singly Linked List of Books). The client 
* interacts only with the 'Iterator' interface (first, next, isDone, current) 
* without knowing how the Books are stored in memory.
* 
* ============================================================================
*/
/**
* ============================================================================
* File: Iterator.cpp (Classic GoF Version)
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the classic Object-Oriented Iterator pattern.
* We build a custom collection (a Singly Linked List of Books). The client 
* interacts only with the 'Iterator' interface (first, next, isDone, current) 
* without knowing how the Books are stored in memory.
* 
* --- MULTIPLE TRAVERSALS:
* Because iterators are separate objects (not a cursor inside the collection), 
* we can instantiate multiple iterators to traverse the same collection 
* simultaneously at different speeds or starting points.
*
* --- CLONING CAPABILITY:
* To match the efficiency of modern C++ iterators, this implementation 
* includes a 'clone()' method. This allows creating a new iterator at 
* the exact same position as an existing one, facilitating efficient 
* nested traversals.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <unordered_set>
#include <vector>
//--------------------------------------------------------- Domain Object:
class Book
{
private:
std::string title_;
std::string author_;
Book()                       = delete;
// 1 DC: Default Constructor
Book(const Book&)            = delete;
// 2 CC: Copy Constructor
Book& operator=(const Book&) = delete;
// 4 CA: Copy Assignment
Book& operator=(Book&&)      = delete;
// 5 MA: Move Assignment
public:
Book(Book&&)                 = default;
// 3 MC: Move Constructor
~Book()                      = default;
// 6 De: Destructor
Book(std::string title, std::string author)
// 7 PC: Parametric Constructor
: title_{std::move(title)}, author_{std::move(author)} { }
std::string getTitle()  const { return title_; }
std::string getAuthor() const { return author_; }
};
//--------------------------------------------------------- Iterator Interface:
template <typename T>
class Iterator
{
public:
virtual ~Iterator()                  = default;
virtual void first()                 = 0;
virtual void next()                  = 0;
virtual bool isDone()          const = 0;
virtual const T& currentBook() const = 0;
// Prototype-like method to duplicate the iterator state
virtual std::unique_ptr<Iterator<T>> clone() const = 0;
};
//--------------------------------------------------------- Internal Node:
class Node
{
public:
Book book;
std::unique_ptr<Node> next{nullptr};
explicit Node(Book book) : book{std::move(book)} { }
};
//--------------------------------------------------------- Concrete Aggregate:
class BookCollection
{
private:
std::unique_ptr<Node> head_;
Node* tail_{nullptr}; 
public:
void addBook(const std::string& title, const std::string& author)
{
auto newNode = std::make_unique<Node>(Book{title, author});
Node* currentNode = newNode.get();
if(!head_) head_ = std::move(newNode);
else       tail_->next = std::move(newNode);
tail_ = currentNode;
}
const Node* getHead() const { return head_.get(); }
std::unique_ptr<Iterator<Book>> createIterator() const;
};
//--------------------------------------------------------- Concrete Iterator:
class BookIterator : public Iterator<Book>
{
private:
const BookCollection& bookCollection_;
const Node* currentNode_;
public:
explicit BookIterator(const BookCollection& bookCollection) 
: bookCollection_{bookCollection}, currentNode_{bookCollection.getHead()} { }
// Internal constructor for cloning
BookIterator(const BookCollection& col, const Node* curr) 
: bookCollection_{col}, currentNode_{curr} { }
void first() override { currentNode_ = bookCollection_.getHead(); }
void next() override
{
if(currentNode_) currentNode_ = currentNode_->next.get();
}
bool isDone() const override { return currentNode_ == nullptr; }
const Book& currentBook() const override
{
if(isDone()) throw std::out_of_range("Iterator is out of bounds");
return currentNode_->book;
}
std::unique_ptr<Iterator<Book>> clone() const override
{
return std::make_unique<BookIterator>(bookCollection_, currentNode_);
}
};
std::unique_ptr<Iterator<Book>> BookCollection::createIterator() const
{
return std::make_unique<BookIterator>(*this);
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== ITERATOR PATTERN (CLASSIC GOF) ===\n" << std::endl;
BookCollection bookCollection;
bookCollection.addBook("The C++ Programming Language", "Bjarne Stroustrup");
bookCollection.addBook("Design Patterns", "Gang of Four");
bookCollection.addBook("Clean Code", "Robert C. Martin");
bookCollection.addBook("The Pragmatic Programmer", "Robert C. Martin");
bookCollection.addBook("Effective C++", "Scott Meyers");
bookCollection.addBook("Clean Agile: Back to Basics", "Robert C. Martin");
bookCollection.addBook("Effective Modern C++", "Scott Meyers");
// --- Test 1: Standard Traversal ---
std::cout << "--- bookCollection Inventory ---\n";
auto iterator = bookCollection.createIterator();
for(iterator->first(); !iterator->isDone(); iterator->next())
{
const Book& book = iterator->currentBook();
std::cout << " - " << book.getTitle() << " (by " << book.getAuthor() << ")\n";
}
// --- Test 2: Multiple Traversals (Finding duplicates) ---
// We will use two independent iterators to find books by the same author.
std::cout << "\n--- Finding multiple books by the same author ---\n";
std::unordered_set<std::string> processedAuthors;
auto outerIt = bookCollection.createIterator();
for(outerIt->first(); !outerIt->isDone(); outerIt->next())
{
const Book& currentBook = outerIt->currentBook();
if(!processedAuthors.contains(currentBook.getAuthor()))
{
std::vector<std::string> bookTitlesOfSameAuthor;
processedAuthors.insert(currentBook.getAuthor());
bookTitlesOfSameAuthor.push_back(currentBook.getTitle());
// Use cloning to start the inner iterator at the same position
auto innerIt = outerIt->clone();
innerIt->next();
// Move one step ahead
while(!innerIt->isDone())
{
const Book& compareBook = innerIt->currentBook();
if(currentBook.getAuthor() == compareBook.getAuthor())
bookTitlesOfSameAuthor.push_back(compareBook.getTitle());
innerIt->next();
}
if(bookTitlesOfSameAuthor.size() > 1)
{
std::cout << "Several books found by " << currentBook.getAuthor() << ":\n";
for(const auto& title : bookTitlesOfSameAuthor)
std::cout << " - " << title << std::endl;
std::cout << std::endl;
}
}
// if
}
// for
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Iterator.cpp (Modern STL / Idiomatic C++ Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* In modern C++, the Iterator pattern is built into the language itself.
* Instead of using virtual interfaces (first, next, isDone), we use 
* Operator Overloading (*, ++, !=) and Duck Typing (begin, end).
* 
* --- THE "RANGE-BASED FOR" MAGIC:
* Because our BookCollection provides begin() and end(), and our BookIterator 
* implements the required operators, we can use the elegant C++11 loop:
*    for(const auto& book : library) { ... }
* 
* This approach has ZERO virtual overhead and provides maximum performance.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <unordered_set>
#include <vector>
//--------------------------------------------------------- Domain Object:
class Book
{
private:
std::string title_;
std::string author_;
Book()                       = delete;
// 1 DC: Default Constructor
Book(const Book&)            = delete;
// 2 CC: Copy Constructor
Book& operator=(const Book&) = delete;
// 4 CA: Copy Assignment
Book& operator=(Book&&)      = delete;
// 5 MA: Move Assignment
public:
Book(Book&&)                 = default;
// 3 MC: Move Constructor
~Book()                      = default;
// 6 De: Destructor
Book(std::string title, std::string author)
// 7 PC: Parametric Constructor
: title_{std::move(title)}, author_{std::move(author)} { }
std::string getTitle()  const { return title_; }
std::string getAuthor() const { return author_; }
};
//--------------------------------------------------------- Internal Node:
class Node
{
public:
Book book;
std::unique_ptr<Node> next;
explicit Node(Book book) : book{std::move(book)}, next{nullptr} { }
};
//--------------------------------------------------------- STL-Style Iterator:
// We don't inherit from any interface. We just provide the expected operators.
class BookIterator
{
private:
const Node* currentNode_;
public:
// The iterator acts as a lightweight pointer wrapper
explicit BookIterator(const Node* node) : currentNode_{node} { }
// 1. Dereference operator (replaces currentItem())
const Book& operator*() const
{
return currentNode_->book;
}
// 2. Pre-increment operator (replaces next())
BookIterator& operator++()
{
if(currentNode_) currentNode_ = currentNode_->next.get();
return *this;
}
// 3. Inequality operator (replaces !isDone())
bool operator!=(const BookIterator& other) const
{
return currentNode_ != other.currentNode_;
}
};
//--------------------------------------------------------- Concrete Aggregate:
class BookCollection
{
private:
std::unique_ptr<Node> head_;
Node* tail_{nullptr}; 
public:
void addBook(const std::string& title, const std::string& author)
{
auto newNode = std::make_unique<Node>(Book{title, author});
Node* current = newNode.get();
if(!head_) head_ = std::move(newNode);
else       tail_->next = std::move(newNode);
tail_ = current;
}
// --- The magic methods for C++ Range-based for loops ---
BookIterator begin() const
{
return BookIterator{head_.get()};
}
BookIterator end() const
{
return BookIterator{nullptr};
// The "done" state is a null pointer
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== ITERATOR PATTERN (IDIOMATIC C++ / STL) ===\n" << std::endl;
BookCollection bookCollection;
bookCollection.addBook("The C++ Programming Language", "Bjarne Stroustrup");
bookCollection.addBook("Design Patterns", "Gang of Four");
bookCollection.addBook("Clean Code", "Robert C. Martin");
bookCollection.addBook("The Pragmatic Programmer", "Robert C. Martin");
bookCollection.addBook("Effective C++", "Scott Meyers");
bookCollection.addBook("Clean Agile: Back to Basics", "Robert C. Martin");
bookCollection.addBook("Effective Modern C++", "Scott Meyers");
// --- Test 1: Standard Traversal (The modern way) ---
std::cout << "--- BookCollection Inventory (Range-based for loop) ---\n";
// This single line replaces: for(iterator->first(); !iterator->isDone(); iterator->next())
//                            {
//                               Book book = iterator->currentItem();
for(const auto& book : bookCollection)
std::cout << " - " << book.getTitle() << " (by " << book.getAuthor() << ")\n";
// --- Test 2: Multiple Traversals (Finding duplicates) ---
// We will use two independent iterators to find books by the same author.
std::cout << "\n--- Finding multiple books by the same author ---\n";
std::unordered_set<std::string> processedAuthors;
// We can still use iterators manually if we need fine-grained control
for(auto outerIt = bookCollection.begin(); outerIt != bookCollection.end(); ++outerIt)
{
const Book& currentBook = *outerIt;
if(!processedAuthors.contains(currentBook.getAuthor()))
{
std::vector<std::string> bookTitlesOfSameAuthor;
processedAuthors.insert(currentBook.getAuthor());
bookTitlesOfSameAuthor.push_back(currentBook.getTitle());
// Start inner iterator one step ahead of the outer iterator
auto innerIt = outerIt;
++innerIt;
while(innerIt != bookCollection.end())
{
const Book& compareBook = *innerIt;
if(currentBook.getAuthor() == compareBook.getAuthor())
bookTitlesOfSameAuthor.push_back(compareBook.getTitle());
++innerIt;
}
if(bookTitlesOfSameAuthor.size() > 1)
{
std::cout << "Several books found by " << currentBook.getAuthor() << ":\n";
for(const auto& title : bookTitlesOfSameAuthor)
std::cout << " - " << title << std::endl;
std::cout << std::endl;
}
}
// if
}
// for
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Mediator.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Mediator Pattern. It evaluates the math 
* expression: (1/a)*(b+c) + d
* 
* Since in this example, math operations are assumed to be expensive, the
* 'Expression' (Mediator) acts as a smart cache coordinator. When a Number
* (Colleague)  changes its value, it notifies the Mediator. The Mediator then
* recalculates ONLY the parts of the expression affected by that specific variable.
* 
* --- MODERN C++:
* - The Mediator identifies which Colleague sent the notification by simply 
*   comparing memory addresses (e.g., if colleague == a_), which is the 
*   fastest and safest way to handle identity in C++.
* ============================================================================
*/
#include <iostream>
#include <exception>
#include <stdexcept>
class Colleague;
//--------------------------------------------------------- Mediator Interface:
class Mediator
{
public:
virtual ~Mediator() = default;
// Communication channel from Colleague to Mediator
virtual void changed(Colleague* colleague) = 0;
};
//--------------------------------------------------------- Colleague Base:
class Colleague
{
protected:
Mediator* mediator_;
// Not owner, safe since lifespans are tied in the stack
explicit Colleague(Mediator* mediator) : mediator_{mediator} { }
void notify() 
{
if(mediator_) mediator_->changed(this);
}
public:
virtual ~Colleague() = default;
};
//------------------------------------------------ Double (Concrete Colleague):
class Double : public Colleague
{
private:
double val_;
public:
Double(Mediator* mediator, double val) : Colleague{mediator}, val_{val} { }
void setVal(double newVal)
{
std::cout << " [Variable] Double changed to " << newVal << '\n';
val_ = newVal;
notify();
// Must be the last line to inform the Mediator
}
double getVal() const { return val_; }
};
//----------------------------------------------- Integer (Concrete Colleague):
class Integer : public Colleague
{
private:
int val_;
public:
Integer(Mediator* mediator, int val) : Colleague{mediator}, val_{val} { }
void setVal(int newVal)
{
std::cout << " [Variable] Integer changed to " << newVal << '\n';
val_ = newVal;
notify();
// Must be the last line to inform the Mediator
}
int getVal() const { return val_; }
};
//--------------------------------------------- Expression (Concrete Mediator):
class Expression : public Mediator
// (1/a)*(b+c) + d
{
private:
// Pointers to Concrete Colleagues
Double*  a_{nullptr};
Double*  b_{nullptr};
Integer* c_{nullptr};
Integer* d_{nullptr};
// Cache and flags for lazy evaluation
bool doInv_{true}, doMul_{true}, doAd1_{true}, doAd2_{true};
double invVal_{0}, mulVal_{0}, ad1Val_{0}, ad2Val_{0};
bool verify() const 
{
return a_ != nullptr && b_ != nullptr && c_ != nullptr && d_ != nullptr;
}
public:
// Type-safe registration methods (replaces dynamic_cast and switch)
void setA(Double* a)  { a_ = a; }
void setB(Double* b)  { b_ = b; }
void setC(Integer* c) { c_ = c; }
void setD(Integer* d) { d_ = d; }
// The Mediator coordinates the cache invalidation based on memory identity
void changed(Colleague* colleague) override
{
if     (colleague == a_)                    doInv_ = doMul_ = doAd2_ = true;
else if(colleague == b_ || colleague == c_) doAd1_ = doMul_ = doAd2_ = true;
else if(colleague == d_)                    doAd2_ = true;
}
// Evaluates (1/a)*(b+c) + d
double evaluate() 
{
if(!verify()) throw std::invalid_argument("Error: Expression not well defined.");
if(doInv_)
{
invVal_ = 1.0 / a_->getVal(); 
std::cout << "\t\tInverting (1/a)\n";
}
if(doAd1_)
{
ad1Val_ = b_->getVal() + c_->getVal(); 
std::cout << "\t\tAdding (b+c)\n";
}
if(doMul_)
{
mulVal_ = invVal_ * ad1Val_; 
std::cout << "\t\tMultiplying\n";
}
if(doAd2_)
{
ad2Val_ = mulVal_ + d_->getVal(); 
std::cout << "\t\tAdding (+d)\n";
}
// Reset flags after evaluation
doInv_ = doMul_ = doAd1_ = doAd2_ = false;
std::cout << "\tResult = ";
return ad2Val_;
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MEDIATOR PATTERN SIMULATION ===\n" << std::endl;
try
{
Expression expression;
// (1/a)*(b+c) + d
// 1. Create Colleagues
Double  a{&expression, 2.0};
Double  b{&expression, 8.0};
Integer c{&expression, 3};
Integer d{&expression, 1};
// 2. Type-safe registration with the Mediator
expression.setA(&a);
expression.setB(&b);
expression.setC(&c);
expression.setD(&d);
std::cout << "\tInitial Evaluation:\n";
std::cout << expression.evaluate() << "\n\n";
std::cout << "\tSecond Evaluation (Cache hit, no operations should run):\n";
std::cout << expression.evaluate() << "\n\n";
a.setVal(4.4);
std::cout << "\tEvaluating after 'a' changes:\n";
std::cout << expression.evaluate() << "\n\n";
c.setVal(10);
std::cout << "\tEvaluating after 'c' changes:\n";
std::cout << expression.evaluate() << "\n\n";
d.setVal(6);
std::cout << "\tEvaluating after 'd' changes:\n";
std::cout << expression.evaluate() << "\n\n";
a.setVal(3.1); d.setVal(4);
std::cout << "\tEvaluating after 'a' and 'c' changes:\n";
std::cout << expression.evaluate() << "\n";
}
catch(const std::exception& e)
{
std::cerr << e.what() << std::endl;
}
}
//================================================================================ END
/**
* ============================================================================
* File: Memento.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the classic GoF Memento pattern.
* 1. Originator: Creates and restores its state from a Memento.
* 2. Memento: An opaque object for everyone except its Creator.
* 3. Caretaker: Manages the history (stack) of Mementos without inspecting.
* 
* --- ENCAPSULATION NOTE:
* By nesting the concrete Memento struct inside the Originator class, 
* we guarantee that ONLY the specific Originator knows the internal 
* structure of its own Memento. The Caretaker only sees a generic 
* base pointer (std::unique_ptr<Memento>).
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <stack>
#include <stdexcept>
//------------------------------------------------------------ Memento Interface:
// Opaque interface for the Caretaker
class Memento
{
public:
virtual ~Memento() = default;
};
using MementoPtr = std::unique_ptr<Memento>;
//--------------------------------------------------------- Originator Interface:
class Originator
{
public:
virtual ~Originator()                                = default;
virtual MementoPtr createMemento()             const = 0;
virtual void restoreMemento(MementoPtr memento)      = 0;
};
//-------------------------------------------------------------------- Caretaker:
// Manages the safekeeping of Mementos without inspecting them.
class Caretaker
{
private:
std::stack<MementoPtr> history_;
public:
void save(MementoPtr memento)
{
history_.push(std::move(memento));
std::cout << "          ... Memento saved to Caretaker.\n";
}
MementoPtr undo()
{
if(history_.empty()) throw std::out_of_range("No more states to undo.");
MementoPtr memento = std::move(history_.top());
history_.pop();
return memento;
}
void discardLatest()
{
if(!history_.empty()) history_.pop();
}
};
//--------------------------------------------------------- Concrete Originator A:
class ComponentA : public Originator
{
private:
std::string state_;
// The concrete Memento is hidden inside the Originator
struct MementoA : public Memento
{
std::string savedState;
explicit MementoA(std::string s) : savedState{std::move(s)} { }
};
public:
void setState(std::string s)
{
state_ = std::move(s);
std::cout << " [Action] Component A (string) set to \"" << state_ << "\"\n";
}
void print() const
{
std::cout << " Current A (string): \"" << state_ << "\"\n";
}
MementoPtr createMemento() const override
{
return std::make_unique<MementoA>(state_);
}
void restoreMemento(MementoPtr memento) override
{
auto* m = dynamic_cast<MementoA*>(memento.get());
if(!m) throw std::invalid_argument("Invalid Memento passed to ComponentA");
std::string oldState = state_;
state_ = std::move(m->savedState);
std::cout << "    -> Component A (string) changed from \"" 
<< oldState << "\" to \"" << state_ << "\"\n";
}
};
//--------------------------------------------------------- Concrete Originator B:
class ComponentB : public Originator
{
private:
int value_{0};
struct MementoB : public Memento
{
int savedValue;
explicit MementoB(int v) : savedValue{v} { }
};
public:
void setValue(int v)
{
value_ = v;
std::cout << " [Action] Component B (integer) set to " << value_ << "\n";
}
void print() const
{
std::cout << " Current B (integer): " << value_ << "\n";
}
MementoPtr createMemento() const override
{
return std::make_unique<MementoB>(value_);
}
void restoreMemento(MementoPtr memento) override
{
auto* m = dynamic_cast<MementoB*>(memento.get());
if(!m) throw std::invalid_argument("Invalid Memento passed to ComponentB");
int oldValue = value_;
value_ = m->savedValue;
std::cout << "    -> Component B (integer) changed from " 
<< oldValue << " to " << value_ << "\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MEMENTO PATTERN (CLASSIC GOF) ===\n" << std::endl;
Caretaker caretaker;
ComponentA a;
ComponentB b;
std::cout << "--- PHASE 1: DOING ACTIONS AND SAVING MEMENTOS ---\n";
a.setState("1st value"); 
caretaker.save(a.createMemento());
b.setValue(1); 
caretaker.save(b.createMemento());
std::cout << "\n";
a.setState("2nd value"); 
caretaker.save(a.createMemento());
b.setValue(2); 
caretaker.save(b.createMemento());
std::cout << "\n";
a.setState("3rd value"); 
caretaker.save(a.createMemento());
b.setValue(3); 
caretaker.save(b.createMemento());
std::cout << "\n--- CURRENT STATE AFTER ALL ACTIONS ---\n";
a.print();
b.print();
std::cout << "\n--- PHASE 2: UNDOING ACTIONS ---\n";
try 
{
std::cout << " [System] Discarding the latest saved states to perform Undo...\n\n";
caretaker.discardLatest();
// Discards B's 3rd state
caretaker.discardLatest();
// Discards A's 3rd state
std::cout << " [Undo] Restoring Component B (integer)...\n";
b.restoreMemento(caretaker.undo()); 
std::cout << "\n [Undo] Restoring Component A (string)...\n";
a.restoreMemento(caretaker.undo()); 
std::cout << "\n [Undo] Restoring Component B (integer)...\n";
b.restoreMemento(caretaker.undo());
std::cout << "\n [Undo] Restoring Component A (string)...\n";
a.restoreMemento(caretaker.undo());
std::cout << "\n--- CURRENT STATE AFTER ALL UNDOES ---\n";
a.print();
b.print();
}
catch(const std::exception& e)
{
std::cerr << " [Error] " << e.what() << "\n";
}
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Memento.cpp (String Serialization Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Memento pattern using string serialization.
* Instead of polymorphic Memento objects, the state is serialized into 
* simple 'std::string' formats.
* 
* --- PERSISTENCE:
* The Caretaker manages a stack of strings. To simulate real-world usage,
* we dump this stack into a file, clear the memory, and restore the objects
* from the file.
* 
* --- SERIALIZATION LOGIC:
* A simple 'ID:VALOR' protocol is used. To prevent newlines from breaking
* the file format, we escape them before saving.
* ============================================================================
*/
#include <fstream>
#include <iostream>
#include <string>
#include <stack>
#include <memory>
//--------------------------------------------------------- Originator A:
class ComponentA
{
private:
std::string state_;
public:
void setState(std::string s)
{
state_ = std::move(s);
std::cout << " [Action] Component A (string) set to \"" << state_ << "\"\n";
}
void print() const { std::cout << " Current A (string): \"" << state_ << "\"\n"; }
std::string serialize() const { return "A:" + state_; }
void deserialize(const std::string& data)
{
std::string oldState = state_;
state_ = data.substr(2);
std::cout << "    -> Component A (string) changed from \"" 
<< oldState << "\" to \"" << state_ << "\"\n";
}
};
//--------------------------------------------------------- Originator B:
class ComponentB
{
private:
int value_{0};
public:
void setValue(int v)
{
value_ = v;
std::cout << " [Action] Component B (integer) set to " << value_ << "\n";
}
void print() const { std::cout << " Current B (integer): " << value_ << "\n"; }
std::string serialize() const { return "B:" + std::to_string(value_); }
void deserialize(const std::string& data)
{
int oldValue = value_;
value_ = std::stoi(data.substr(2));
std::cout << "    -> Component B (integer) changed from " 
<< oldValue << " to " << value_ << "\n";
}
};
//--------------------------------------------------------- Caretaker:
class Caretaker
{
private:
std::stack<std::string> history_;
public:
void save(const std::string& memento)
{
history_.push(memento);
std::cout << "          ... Memento (string) saved to Caretaker.\n";
}
std::string undo()
{
if(history_.empty()) throw std::runtime_error("No more states to undo.");
std::string memento = history_.top();
history_.pop();
return memento;
}
void saveToFile(const std::string& filename) const
{
std::ofstream ofs(filename);
std::stack<std::string> tmpStraight = history_;
std::stack<std::string> tmpInverse;
while(!tmpStraight.empty())
// Reverse
{
tmpInverse.push(tmpStraight.top());
tmpStraight.pop();
}
while(!tmpInverse.empty())
// Save to file
{
ofs << tmpInverse.top() << "\n";
tmpInverse.pop();
}
std::cout << " [System] Stack saved to file: " << filename << "\n";
}
void restoreFromFile(const std::string& filename)
{
std::ifstream ifs(filename);
std::string line;
while(std::getline(ifs, line)) if(!line.empty()) save(line);
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MEMENTO PATTERN (STRING SERIALIZATION) ===\n" << std::endl;
std::unique_ptr<Caretaker> caretaker = std::make_unique<Caretaker>();
ComponentA a;
ComponentB b;
std::cout << "--- PHASE 1: DOING ACTIONS AND SAVING ---\n";
a.setState("1st value"); caretaker->save(a.serialize());
b.setValue(1);           caretaker->save(b.serialize());
a.setState("2nd value"); caretaker->save(a.serialize());
b.setValue(2);           caretaker->save(b.serialize());
a.setState("3rd value"); caretaker->save(a.serialize());
b.setValue(3);           caretaker->save(b.serialize());
std::cout << "\n--- CURRENT STATE AFTER ALL ACTIONS ---\n";
a.print();
b.print();
std::cout << "\n--- SAVING CURRENT STATE TO FILE ---\n";
caretaker->saveToFile("memento_stack_data.txt");
std::cout << "\n--- PHASE 2: SIMULATING A FRESH RESTART ---\n";
std::cout << " [System] Clearing memory by reinstantiating a new caretaker..." << std::endl;
caretaker = std::make_unique<Caretaker>();
// Create a new caretaker destroying the old one
std::cout << "\n--- RESTORING FROM DISK ---\n";
caretaker->restoreFromFile("memento_stack_data.txt");
std::cout << "\n--- PERFORMING UNDOES ---\n";
caretaker->undo();
// Discards B's 3rd state
caretaker->undo();
// Discards A's 3rd state
std::cout << " [Undo] Restoring Component B (integer)...\n";
b.deserialize(caretaker->undo()); 
std::cout << "\n [Undo] Restoring Component A (string)...\n";
a.deserialize(caretaker->undo()); 
std::cout << "\n [Undo] Restoring Component B (integer)...\n";
b.deserialize(caretaker->undo()); 
std::cout << "\n [Undo] Restoring Component A (string)...\n";
a.deserialize(caretaker->undo()); 
std::cout << "\n--- CURRENT STATE AFTER ALL UNDOES ---\n";
a.print();
b.print();
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Memento.cpp (Persistent Mmap with Auto-Save & Crash Simulation)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates a robust Memento pattern with disk persistence.
* 1. Auto-Save: Components notify the Caretaker of every change.
* 2. Crash Recovery: The Caretaker restores the system state from the 
*    binary file (".bin") upon instantiation.
* 3. Time Travel: Full Undo/Redo support. New actions after an Undo 
*    will overwrite any existing "future" history.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdexcept>
#include <memory>
//--------------------------------------------------------- Binary Memento (POD):
struct FileHeader
{
size_t count;
// Total valid states in the timeline // (8 bytes)
size_t cursor;
// Current position in time           // (8 bytes), Total = 16 bytes
};
struct SystemState
{
int valB;
// ( 4 bytes)
char strA[64];
// (64 bytes), Total = 68 bytes
};
//--------------------------------------------------------- Concrete Originators:
class Caretaker;
// Forward declaration
class ComponentA
{
private:
std::string state_{""};
Caretaker& caretaker_;
public:
ComponentA(Caretaker& c) : caretaker_{c} { }
void setState(std::string s);
// Defined after Caretaker
void internalSet(std::string s) { state_ = std::move(s); }
void print() const { std::cout << " Current A (string):  \"" << state_ << "\"\n"; }
std::string getState() const { return state_; }
};
class ComponentB
{
private:
int value_{0};
Caretaker& caretaker_;
public:
ComponentB(Caretaker& c) : caretaker_{c} { }
void setValue(int v);
// Defined after Caretaker
void internalSet(int v) { value_ = v; }
void print() const { std::cout << " Current B (integer): " << value_ << "\n"; }
int getValue() const { return value_; }
};
//--------------------------------------------------------- Persistent Caretaker:
class Caretaker
{
private:
constexpr static size_t MAX_STATES = 120;
// FILE_SIZE = 16 + 68*120 = 8176 almost 8kB
constexpr static size_t FILE_SIZE = sizeof(FileHeader) + (sizeof(SystemState) * MAX_STATES);
int fd_;
void* mappedRegion_;
FileHeader* header_;
SystemState* history_;
ComponentA* a_{nullptr};
ComponentB* b_{nullptr};
bool isRestoring_{false}; 
public:
explicit Caretaker(const std::string& filename)
{
fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0644);
if(fd_ < 0) throw std::runtime_error("Could not open persistence file.");
off_t currentSize = lseek(fd_, 0, SEEK_END);
if(currentSize == 0)
if(ftruncate(fd_, FILE_SIZE) != 0) throw std::runtime_error("File allocation failed.");
mappedRegion_ = mmap(NULL, FILE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
if(mappedRegion_ == MAP_FAILED) throw std::runtime_error("mmap failed.");
header_ = static_cast<FileHeader*>(mappedRegion_);
history_ = reinterpret_cast<SystemState*>((char*)mappedRegion_ + sizeof(FileHeader));
}
~Caretaker()
{
munmap(mappedRegion_, FILE_SIZE);
close(fd_);
}
void setComponents(ComponentA* a, ComponentB* b)
{
a_ = a; b_ = b;
if(header_->count > 0)
{
std::cout << " [System] Recovery: Restoring state " << header_->cursor << " from disk.\n";
applyState(history_[header_->cursor]);
}
}
void save()
{
if(isRestoring_) return;
if(header_->cursor + 1 >= MAX_STATES) 
throw std::runtime_error("History file is full! Cannot save more states.\n");
SystemState systemSate;
systemSate.valB = b_->getValue();
std::strncpy(systemSate.strA, a_->getState().c_str(), 63);
systemSate.strA[63] = '\0';
if(header_->count > 0) header_->cursor++;
history_[header_->cursor] = systemSate;
header_->count = header_->cursor + 1;
std::cout << "          ... Automatic checkpoint saved (State " << header_->cursor << ").\n";
}
void undo()
{
if(header_->cursor > 0)
{
isRestoring_ = true;
header_->cursor--;
applyState(history_[header_->cursor]);
std::cout << " [Undo] System rolled back to state " << header_->cursor << "\n";
isRestoring_ = false;
}
else std::cout << " [System] Cannot Undo: start of history reached.\n";
}
void redo()
{
if(header_->cursor < header_->count - 1)
{
isRestoring_ = true;
header_->cursor++;
applyState(history_[header_->cursor]);
std::cout << " [Redo] System moved forward to state " << header_->cursor << "\n";
isRestoring_ = false;
}
else std::cout << " [System] Cannot Redo: latest state reached.\n";
}
void applyState(const SystemState& s)
{
a_->internalSet(s.strA);
b_->internalSet(s.valB);
}
void resetFile()
{
header_->count = 0;
header_->cursor = 0;
}
};
//--------------------------------------------------------- Implementation:
void ComponentA::setState(std::string s) 
{ 
state_ = std::move(s); 
std::cout << " [Action] Component A set to \"" << state_ << "\"\n";
caretaker_.save(); 
}
void ComponentB::setValue(int v) 
{ 
value_ = v; 
std::cout << " [Action] Component B set to " << value_ << "\n";
caretaker_.save(); 
}
//--------------------------------------------------------- Main Simulation:
int main()
{
try
{
std::cout << "=== MEMENTO PATTERN (MMAP PERSISTENCE & CRASH SIMULATION) ===\n" << std::endl;
const std::string dbFile = "memento_history.bin";
{
std::cout << "--- PHASE 1: INITIAL EXECUTION & SAVE ---\n";
auto caretaker = std::make_unique<Caretaker>(dbFile);
caretaker->resetFile(); 
ComponentA a(*caretaker);
ComponentB b(*caretaker);
caretaker->setComponents(&a, &b);
a.setState("Alpha");
// A="Alpha", B=0   (State 0)
a.setState("Beta");
// A="Beta",  B=0   (State 1)
b.setValue(50);
// A="Beta",  B=50  (State 2)
b.setValue(100);
// A="Beta",  B=100 (State 3)
a.print(); b.print();
std::cout << " [CRASH] Program terminated unexpectedly!\n\n";
} 
{
std::cout << "--- PHASE 2: RESTART & RECOVERY ---\n";
auto caretaker = std::make_unique<Caretaker>(dbFile);
ComponentA a(*caretaker);
ComponentB b(*caretaker);
caretaker->setComponents(&a, &b);
// Recovers State 3 (A="Beta", B=100)
a.print(); b.print();
std::cout << "\n--- PHASE 3: UNDO & REDO ---\n";
caretaker->undo();
// A="Beta",  B=50  (State 2)
a.print(); b.print();
caretaker->redo();
// A="Beta",  B=100 (State 3)
a.print(); b.print();
std::cout << "\n--- PHASE 4: CONTINUING ACTIONS ---\n";
a.setState("Gamma");
// A="Gamma", B=100 (State 4)
b.setValue(200);
// A="Gamma", B=200 (State 5)
b.setValue(800);
// A="Gamma", B=800 (State 6)
a.print(); b.print();
std::cout << "\n--- PHASE 5: MULTIPLE UNDOS ---\n";
caretaker->undo();
// A="Gamma", B=200 (State 5)
a.print(); b.print();
caretaker->undo();
// A="Gamma", B=100 (State 4)
a.print(); b.print();
caretaker->undo();
// A="Beta",  B=100 (State 3)
a.print(); b.print();
std::cout << "\n--- PHASE 6: REDO TEST ---\n";
caretaker->redo();
// A="Gamma", B=100 (State 4)
a.print(); b.print();
std::cout << "\n--- PHASE 7: NEW ACTION (KILLING REDO) ---\n";
b.setValue(600);
// A="Gamma", B=600 (New State 5, old States 5 & 6 are lost)
a.print(); b.print();
std::cout << "\n--- PHASE 8: ATTEMPTING TO REDO (SHOULD FAIL) ---\n";
caretaker->redo();
a.print(); b.print();
std::cout << "\n--- PHASE 9: UNDO & REDO ---\n";
caretaker->undo();
// A="Gamma", B=100 (State 4)
a.print(); b.print();
caretaker->redo();
// A="Gamma", B=600 (New State 5)
a.print(); b.print();
}
}
catch(const std::runtime_error& e)
{
std::cerr << "\nCaught error: " << e.what() << std::endl;
}
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Observer.cpp (Classic GoF Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation follows the classic Gang of Four (GoF) approach. 
* It uses a polymorphic base class 'Observable' and 'Observer'. 
* The Observable maintains a collection of Observers and notifies them 
* via a single 'update(Observable*)' method.
* 
* --- THE GOF TRADE-OFF:
* Since the 'update' method accepts a base pointer, the ConcreteObserver 
* must cast this pointer to the concrete type (e.g., 'NumberObservable') to
* access specific data. This is the traditional way to implement the pattern
* before modern C++ techniques like CRTP were popularized.
* 
* --- ARCHITECTURE:
* 1. Observable: Manages the list of Observers.
* 2. Observer: Defines the interface for update notifications.
* 3. Concrete Implementation: Observers query the Observable to synchronize.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
//--------------------------------------------------------- Observer Interface:
class Observable;
// Forward declaration
class Observer
{
public:
virtual ~Observer() = default;
virtual void update(Observable* observable) = 0;
};
//--------------------------------------------------------- Observable Base:
class Observable
{
private:
std::vector<Observer*> observers_;
public:
virtual ~Observable() = default;
void attach(Observer* obs)
{
observers_.push_back(obs);
}
void detach(Observer* obs)
{
observers_.erase(std::remove(observers_.begin(), observers_.end(), obs), 
observers_.end());
}
protected:
void notify()
{
for (auto* obs : observers_) obs->update(this);
}
};
//--------------------------------------------------------- Concrete Observable:
class NumberObservable : public Observable
{
private:
int val_{0};
public:
void setVal(int newVal)
{
val_ = newVal;
notify();
}
int getVal() const { return val_; }
};
//--------------------------------------------------------- Concrete Observers:
class DivObserver : public Observer
{
private:
int div_;
public:
explicit DivObserver(int div) : div_{div} { }
void update(Observable* obs) override
{
// The classic GoF way: cast the base pointer to the concrete type
auto* numberObservable = static_cast<NumberObservable*>(obs);
int n = numberObservable->getVal();
std::cout << "  [DivObserver] " << n << " div " << div_ << " is " << n / div_ << "\n";
}
};
class ModObserver : public Observer
{
private:
int div_;
public:
explicit ModObserver(int div) : div_{div} { }
void update(Observable* obs) override
{
auto* numberObservable = static_cast<NumberObservable*>(obs);
int n = numberObservable->getVal();
std::cout << "  [ModObserver] " << n << " mod " << div_ << " is " << n % div_ << "\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== OBSERVER PATTERN (CLASSIC GOF) ===\n" << std::endl;
NumberObservable numberObservable;
// One Observable
DivObserver divObserver{4};
// Two Observers
ModObserver modObserver{3};
numberObservable.attach(&divObserver);
numberObservable.attach(&modObserver);
std::cout << "--- Changing numberObservable state to 14 (with two observers):\n";
numberObservable.setVal(14);
std::cout << "\n--- Detaching divObserver and changing numberObservable to 20:\n";
numberObservable.detach(&divObserver);
numberObservable.setVal(20);
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Observer.cpp (Without Topics)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation uses the CRTP (Curiously Recurring Template Pattern) 
* to inject the concrete observable type into the base Observable and 
* Observer classes. This allows the Observer to interact directly with 
* the specific Observable without needing a dynamic_cast.
* 
* --- CROSS-REFERENCE (CRTP):
* For a deep dive into how Static Polymorphism and the Curiously Recurring 
* Template Pattern (CRTP) work in C++, please see folder '32_CRTP/'.
* 
* --- MULTIPLE OBSERVABLES:
* This implementation supports multiple Observables for each Observer. 
* The Observable passed to the 'update' operation allows the Observer 
* to interact with the specific Observable that has changed.
* 
* --- TERMINOLOGY:
* We use 'Observable' instead of the classic GoF term 'Subject' because 
* it describes the role of the watched object much more clearly.
* ============================================================================
*/
#include <iostream>
#include <set>
#include <string>
//--------------------------------------------------------- Observer Interface:
template<class ConcreteObservable>
class Observer
{
private:
template<class> friend class Observable;
protected:
virtual ~Observer() = default;
// Called by the Observable when its state changes
virtual void update(ConcreteObservable&) = 0;
};
//--------------------------------------------------------- Observable Base:
// Uses CRTP to know its own concrete type at compile time
template<class ConcreteObservable>
class Observable
{
private:
// Non-owning pointers. Observers must outlive or detach themselves.
std::set<Observer<ConcreteObservable>*> observers_;
// Force constructor to be private to ensure only ConcreteObservable can build it
Observable() = default;
friend ConcreteObservable;
protected:
virtual ~Observable() = default;
void notify()
{
for (auto obs : observers_) obs->update(*static_cast<ConcreteObservable*>(this));
}
public:
void attach(Observer<ConcreteObservable>& obs) { observers_.insert(&obs); }
void detach(Observer<ConcreteObservable>& obs) { observers_.erase(&obs); }
};
//--------------------------------------------------------- Concrete Observable:
class NumberObservable : public Observable<NumberObservable>
{
private:
int val_{0};
public:
void setVal(int i)
{
val_ = i;
notify();
// Must be the last line of the state change
}
int getVal() const { return val_; }
};
//--------------------------------------------------------- Concrete Observers:
class DivObserver : public Observer<NumberObservable>
{
private:
int div_;
protected:
void update(NumberObservable& num) override
{
int n = num.getVal();
std::cout << "  [DivObserver] " << n << " div " << div_ << " is " << n / div_ << '\n';
}
public:
explicit DivObserver(int div) : div_{div} { }
};
class ModObserver : public Observer<NumberObservable>
{
private:
int div_;
protected:
void update(NumberObservable& num) override
{
int n = num.getVal();
std::cout << "  [ModObserver] " << n << " mod " << div_ << " is " << n % div_ << '\n';
}
public:
explicit ModObserver(int div) : div_{div} { }
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== OBSERVER PATTERN (WITHOUT TOPICS) ===\n" << std::endl;
// Create two observables
NumberObservable numberObservable_1;
NumberObservable numberObservable_2;
// Create three observers
DivObserver observer_1{4};
DivObserver observer_2{3};
ModObserver observer_3{3};
// Attach observers to observables (demonstrating multiple Observables per Observer)
numberObservable_1.attach(observer_1);
numberObservable_1.attach(observer_2);
numberObservable_1.attach(observer_3);
numberObservable_2.attach(observer_1);
numberObservable_2.attach(observer_2);
numberObservable_2.attach(observer_3);
std::cout << "--- With three observers attached to numberObservables 1 and 2:\n";
std::cout << "Setting numberObservable_1 to 14:\n";
numberObservable_1.setVal(14);
std::cout << "\nSetting numberObservable_2 to 18:\n";
numberObservable_2.setVal(18);
std::cout << "\n--- Detaching observer_2 from numberObservable_1:\n";
numberObservable_1.detach(observer_2);
std::cout << "Setting numberObservable_1 to 14 again:\n";
numberObservable_1.setVal(14);
std::cout << "\n--- Detaching observers 1 and 3, and reattaching observer_2 to numberObservable_1:\n";
numberObservable_1.detach(observer_1);
numberObservable_1.detach(observer_3);
numberObservable_1.attach(observer_2);
std::cout << "Setting numberObservable_1 to 14 again:\n";
numberObservable_1.setVal(14);
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Observer.cpp (Memory-Safe with Smart Pointers)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation uses the CRTP (Curiously Recurring Template Pattern) 
* to inject the concrete observable type into the base Observable and 
* Observer classes. This allows the Observer to interact directly with 
* the specific Observable without needing a dynamic_cast.
* 
* --- MEMORY SAFETY UPDATE:
* This version replaces raw pointers with std::weak_ptr in the Observable's 
* internal list. This ensures that if an Observer is destroyed, the 
* Observable will not attempt to call a method on a dead object.
* 
* --- CROSS-REFERENCE (CRTP):
* For a deep dive into how Static Polymorphism and the Curiously Recurring 
* Template Pattern (CRTP) work in C++, please see folder '32_CRTP/'.
* 
* --- MULTIPLE OBSERVABLES:
* This implementation supports multiple Observables for each Observer. 
* The Observable passed to the 'update' operation allows the Observer 
* to interact with the specific Observable that has changed.
* 
* --- TERMINOLOGY:
* We use 'Observable' instead of the classic GoF term 'Subject' because 
* it describes the role of the watched object much more clearly.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
//--------------------------------------------------------- Observer Interface:
template<class ConcreteObservable>
class Observer
{
private:
template<class> friend class Observable;
protected:
virtual ~Observer() = default;
// Called by the Observable when its state changes
virtual void update(ConcreteObservable&) = 0;
};
//--------------------------------------------------------- Observable Base:
// Uses CRTP to know its own concrete type at compile time
template<class ConcreteObservable>
class Observable
{
private:
// Now using weak_ptr to track observers without owning them.
// This prevents crashes if an observer is destroyed unexpectedly.
mutable std::vector<std::weak_ptr<Observer<ConcreteObservable>>> observers_;
// Force constructor to be private to ensure only ConcreteObservable can build it
Observable() = default;
friend ConcreteObservable;
protected:
virtual ~Observable() = default;
void notify()
{
auto it = observers_.begin();
while (it != observers_.end()) {
if (auto obs = it->lock()) {
obs->update(*static_cast<ConcreteObservable*>(this));
++it;
} else {
it = observers_.erase(it);
// Automatic cleanup of dead observers
}
}
}
public:
void attach(std::shared_ptr<Observer<ConcreteObservable>> obs)
{ 
observers_.push_back(obs); 
}
void detach(std::shared_ptr<Observer<ConcreteObservable>> obs)
{
observers_.erase(
std::remove_if(observers_.begin(), observers_.end(),
[&obs](const std::weak_ptr<Observer<ConcreteObservable>>& wp) {
return wp.lock() == obs;
}),
observers_.end());
}
};
//--------------------------------------------------------- Concrete Observable:
class NumberObservable : public Observable<NumberObservable>
{
private:
int val_{0};
public:
void setVal(int i)
{
val_ = i;
notify();
// Must be the last line of the state change
}
int getVal() const { return val_; }
};
//--------------------------------------------------------- Concrete Observers:
class DivObserver : public Observer<NumberObservable>
{
private:
int div_;
protected:
void update(NumberObservable& num) override
{
int n = num.getVal();
std::cout << "  [DivObserver] " << n << " div " << div_ << " is " << n / div_ << '\n';
}
public:
explicit DivObserver(int div) : div_{div} { }
};
class ModObserver : public Observer<NumberObservable>
{
private:
int div_;
protected:
void update(NumberObservable& num) override
{
int n = num.getVal();
std::cout << "  [ModObserver] " << n << " mod " << div_ << " is " << n % div_ << '\n';
}
public:
explicit ModObserver(int div) : div_{div} { }
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MODERN SAFE OBSERVER PATTERN ===\n" << std::endl;
// Create two observables
NumberObservable numberObservable_1;
NumberObservable numberObservable_2;
// Create three observers using shared_ptr
auto observer_1 = std::make_shared<DivObserver>(4);
auto observer_2 = std::make_shared<DivObserver>(3);
auto observer_3 = std::make_shared<ModObserver>(3);
// Attach observers to observables (demonstrating multiple Observables per Observer)
numberObservable_1.attach(observer_1);
numberObservable_1.attach(observer_2);
numberObservable_1.attach(observer_3);
numberObservable_2.attach(observer_1);
numberObservable_2.attach(observer_2);
numberObservable_2.attach(observer_3);
std::cout << "--- With three observers attached to numberObservables 1 and 2:\n";
std::cout << "Setting numberObservable_1 to 14:\n";
numberObservable_1.setVal(14);
std::cout << "\nSetting numberObservable_2 to 18:\n";
numberObservable_2.setVal(18);
std::cout << "\n--- Detaching observer_2 from numberObservable_1:\n";
numberObservable_1.detach(observer_2);
std::cout << "Setting numberObservable_1 to 14 again:\n";
numberObservable_1.setVal(14);
std::cout << "\n--- Detaching observers 1 and 3, and reattaching observer_2 to numberObservable_1:\n";
numberObservable_1.detach(observer_1);
numberObservable_1.detach(observer_3);
numberObservable_1.attach(observer_2);
std::cout << "Setting numberObservable_1 to 14 again:\n";
numberObservable_1.setVal(14);
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Observer.cpp (With Topics)
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This advanced implementation introduces "Topics" (or Events). Instead
* of broadcasting every change to all observers, observers subscribe only
* to specific topics they care about (e.g., Value changes vs Color changes).
* This significantly reduces unnecessary updates and improves efficiency.
*
* --- CROSS-REFERENCE (CRTP):
* For a deep dive into Static Polymorphism and the Curiously Recurring
* Template Pattern (CRTP) used here, please see folder '32_CRTP/'.
*
* --- MULTIPLE OBSERVABLES:
* This implementation supports multiple Observables for each Observer.
* The Observable passed to the 'update' operation allows the Observer
* to interact with the specific Observable that has changed.
*
* --- TERMINOLOGY:
* We use 'Observable' instead of the classic GoF term 'Subject' because
* it describes the role of the watched object much more clearly.
* ============================================================================
*/
#include <iostream>
#include <unordered_set>
// In this case is faster than <set>
#include <vector>
#include <type_traits>
//--------------------------------------------------------- Topics (Events):
// We use 'enum class' for modern type safety.
// 'END' must be the last element to properly size the vector of sets.
enum class Topic
{
Value,
Color,
END
// Must be the last topic
};
//--------------------------------------------------------- Forward Declaration:
template<class ConcreteObservable, Topic TopicType>
class Observer;
//--------------------------------------------------------- Observable Base:
template<class ConcreteObservable>
class Observable
{
private:
// A vector of unordered_sets. Each index in the vector represents a Topic.
// We store void* because different topics might have different Observer types.
std::vector<std::unordered_set<void*>> observers_;
Observable()
{
// Resize the vector to accommodate all topics in O(1) access time
observers_.resize(static_cast<size_t>(Topic::END));
}
friend ConcreteObservable;
protected:
virtual ~Observable() = default;
template<size_t Index = 0>
void dispatchNotify(void* obs, Topic t)
{
if constexpr (Index < static_cast<size_t>(Topic::END))
{
if (Index == static_cast<size_t>(t))
{
constexpr Topic TopicValue = static_cast<Topic>(Index);
static_cast<Observer<ConcreteObservable, TopicValue>*>(obs)->update(*static_cast<ConcreteObservable*>(this), t);
}
else
dispatchNotify<Index + 1>(obs, t);
}
}
void notify(Topic t)
{
size_t index = static_cast<size_t>(t);
for (auto obs : observers_[index])
dispatchNotify(obs, t);
}
public:
// Auto-detect and attach based on Observer inheritance
template<class T>
void attach(T& obs)
{
if constexpr (std::is_base_of_v<Observer<ConcreteObservable, Topic::Value>, T>)
observers_[static_cast<size_t>(Topic::Value)].insert(&obs);
if constexpr (std::is_base_of_v<Observer<ConcreteObservable, Topic::Color>, T>)
observers_[static_cast<size_t>(Topic::Color)].insert(&obs);
}
// Detach from all potential topic sets
template<class T>
void detach(T& obs)
{
for (auto& observer_set : observers_) observer_set.erase(&obs);
}
};
//--------------------------------------------------------- Observer Interface:
template<class ConcreteObservable, Topic TopicType>
class Observer
{
template<class> friend class Observable;
protected:
virtual ~Observer() = default;
// The update method receives both the specific Observable and the Topic
virtual void update(ConcreteObservable&, Topic) = 0;
};
//--------------------------------------------------------- Concrete Observable:
class NumberObservable : public Observable<NumberObservable>
{
private:
int  value_{0};
char color_{'A'};
public:
void setValue(int newValue)
{
value_ = newValue;
notify(Topic::Value);
// Notify only subscribers to 'Value'
}
void setColor(char newColor)
{
color_ = newColor;
notify(Topic::Color);
// Notify only subscribers to 'Color'
}
int  getValue() const {return value_;}
char getColor() const {return color_;}
};
//--------------------------------------------------------- Concrete Observers:
// This observer listens to BOTH Value and Color changes.
class ValueColorObserver : public Observer<NumberObservable, Topic::Value>,
public Observer<NumberObservable, Topic::Color>
{
protected:
void update(NumberObservable& num, Topic t) override
{
switch (t)
{
case Topic::Value:
std::cout << "  [ValueColorObserver] Value changed to " << num.getValue() << '\n';
break;
case Topic::Color:
std::cout << "  [ValueColorObserver] Color changed to " << num.getColor() << '\n';
break;
default:
break;
}
}
};
// This observer listens ONLY to Value changes.
class ValueObserver : public Observer<NumberObservable, Topic::Value>
{
protected:
void update(NumberObservable& num, Topic) override
{
std::cout << "  [ValueObserver] Value changed to " << num.getValue() << '\n';
}
};
// This observer listens ONLY to Color changes.
class ColorObserver : public Observer<NumberObservable, Topic::Color>
{
protected:
void update(NumberObservable& num, Topic) override
{
std::cout << "  [ColorObserver] Color changed to " << num.getColor() << '\n';
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== OBSERVER PATTERN (WITH TOPICS) ===\n" << std::endl;
// Create two observables with two topics each (Value and Color)
NumberObservable numberObservable_1;
NumberObservable numberObservable_2;
// Create three observers
ValueColorObserver observer_1;
ValueObserver      observer_2;
ColorObserver      observer_3;
// Attach observers automatically. No need to specify the Topic!
numberObservable_1.attach(observer_1);
numberObservable_1.attach(observer_2);
numberObservable_1.attach(observer_3);
std::cout << "--- Initial state changes for numberObservable_1:\n";
std::cout << "Setting Value to 14 (Triggers ValueColorObserver and ValueObserver):\n";
numberObservable_1.setValue(14);
std::cout << "\nSetting Color to 'A' (Triggers ValueColorObserver and ColorObserver):\n";
numberObservable_1.setColor('A');
std::cout << "\n--- Detaching observer_1 from both topics on numberObservable_1...\n";
numberObservable_1.detach(observer_1);
std::cout << "--- Attaching observer_1 to numberObservable_2...\n";
numberObservable_2.attach(observer_1);
std::cout << "\nSetting Value to 18 on numberObservable_1 (Should only trigger ValueObserver):\n";
numberObservable_1.setValue(18);
std::cout << "\nSetting Color to 'B' on numberObservable_1 (Should only trigger ColorObserver):\n";
numberObservable_1.setColor('B');
std::cout << "\nSetting Value to 15 and Color to 'C' on numberObservable_2 (Triggers ValueColorObserver):\n";
numberObservable_2.setValue(15);
numberObservable_2.setColor('C');
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
// **************************************************************************
// Observer.cpp - Main application demonstrating the Observer Pattern
// Author: Mario Galindo Queralt, Ph.D.
// **************************************************************************
// This program illustrates a modern implementation of the Observer pattern 
// using libsigc++ and C++14/23 smart pointers.
//
// Application Flow:
// 1. Instantiates Subjects ('Number') and Concrete Observers.
// 2. Uses std::make_unique for modern, safe memory management.
// 3. Establishes many-to-many connections between subjects and observers.
// 4. Demonstrates dynamic connection management (attaching/disconnecting).
//
// Technical Note:
// All observers inherit from sigc::trackable, ensuring that they are 
// automatically disconnected from subjects upon destruction, 
// preventing any invalid memory access.
// **************************************************************************
#include "Objects.h"
#include <memory>
// Required for std::unique_ptr and make_unique
int main()
{
Number number_1, number_2;
// Create Observers using std::make_unique
auto observer_1 = std::make_unique<DivObserver>(4);
auto observer_2 = std::make_unique<DivObserver>(3);
auto observer_3 = std::make_unique<ModObserver>(3);
// When using unique_ptr, we pass the dereferenced content (*)
// attach(Observer&) receives the reference and mem_fun handles the rest
sigc::connection con_1 = number_1.attach(*observer_1);
sigc::connection con_2 = number_1.attach(*observer_2);
sigc::connection con_3 = number_1.attach(*observer_3);
number_2.attach(*observer_1);
number_2.attach(*observer_2);
number_2.attach(*observer_3);
std::cout << "With three observers (unique_ptr):\n";
number_1.setVal(14);
number_2.setVal(18);
std::cout << "\nWith two observers (disconnecting con_2):\n";
con_2.disconnect();
number_1.setVal(14);
std::cout << "\nWith one observer (re-attaching observer_2):\n";
con_1.disconnect();
con_3.disconnect();
number_1.attach(*observer_2);
number_1.setVal(14);
// Upon exiting main, the unique_ptrs are automatically destroyed.
// Thanks to sigc::trackable, there will be no invalid access attempts.
}
//================================================================================ END
/**
* ============================================================================
* File: FSM.cpp (Simple Example)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the State pattern using a Finite State Machine (FSM).
* Instead of using complex 'if-else' or 'switch' logic in the main class,
* we move state-specific behavior into separate State classes.
* 
* --- DYNAMIC TRANSITIONS:
* The FSM Context holds a pointer to the current State. When an event occurs,
* the Context delegates the call to the current State object. If a transition
* is required, the State object tells the Context to change to a new State.
* 
* --- AUTOMATIC REGISTRATION:
* Each State class registers itself into the FSM's internal map during
* static initialization, making the machine modular and extensible.
* ============================================================================
*/
#include "FSM.h"
int main()
{
std::cout << "=== FINITE STATE MACHINE (SIMPLE EXAMPLE) ===\n" << std::endl;
FSM fsm1("fsm1", "State_A");
fsm1.event_change();
std::cout << '\n';
FSM fsm2("fsm2", "State_A");
fsm2.event_stay();
fsm2.event_only_A();
fsm2.event_change();
fsm2.event_stay();
try
{
fsm2.event_only_A();
}
catch (const std::exception& e)
{
std::cout << e.what();
}
fsm2.event_change();
fsm2.event_stay();
fsm2.event_only_A();
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
// File: State_A.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class State_A : public IState_FSM
{
public:
// Static registration method that must be called before using the state
static inline bool registered = []()
{
FSM::registerState<State_A>("State_A");
return true;
}();
void onEntry(FSM* fsm) override
{
std::cout << fsm->name << ": State_A::onEntry, i=" << fsm->i++ << "\n";
}
void onExit(FSM* fsm) override
{
std::cout << fsm->name << ": State_A::onExit, i=" << fsm->i++ << "\n\n";
}
void event_stay(FSM* fsm) override
{
std::cout << fsm->name << ": State_A, event_stay, i=" << fsm->i++ << "\n";
}
void event_change(FSM* fsm) override
{
std::cout << fsm->name << ": State_A, event_change, i=" << fsm->i++ << "\n";
fsm->changeState("State_B", &FSM::transition_AtoB);
}
void event_only_A(FSM* fsm) override
{
std::cout << fsm->name << ": State_A, event_only_A, i=" << fsm->i++ << "\n";
}
};
//================================================================================ END
// File: State_B.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class State_B : public IState_FSM
{
public:
// Static registration method
static inline bool registered = []()
{
FSM::registerState<State_B>("State_B");
return true;
}();
void onEntry(FSM* fsm) override
{
std::cout << fsm->name << ": State_B::onEntry, i=" << fsm->i++ << "\n";
}
void onExit(FSM* fsm) override
{
std::cout << fsm->name << ": State_B::onExit, i=" << fsm->i++ << "\n\n";
}
void event_stay(FSM* fsm) override
{
std::cout << fsm->name << ": State_B, event_stay, i=" << fsm->i++ << "\n";
}
void event_change(FSM* fsm) override
{
std::cout << fsm->name << ": State_B, event_change, i=" << fsm->i++ << "\n";
fsm->changeState("State_A", &FSM::transition_BtoA);
}
};
//================================================================================ END
// File: Closed_with_CD.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class Closed_with_CD : public IState_FSM
{
public:
void onEntry(FSM* fsm) override
{
std::cout << fsm->name << ": Turning ON yellow light.\n";
fsm->yellow_light_on = true;
}
void onExit(FSM* fsm) override
{
std::cout << fsm->name << ": Turning OFF yellow light.\n";
fsm->yellow_light_on = false;
}
void open(FSM* fsm) override 
{
fsm->changeState("Open_with_CD", &FSM::open_tray);
}
void play(FSM* fsm) override 
{
fsm->changeState("Playing", &FSM::reset);
}
};
// Auto-register
static inline bool registered_ = []()
{
FSM::registerState<Closed_with_CD>("Closed_with_CD");
return true;
}();
//================================================================================ END
// File: Closed_without_CD.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class Closed_without_CD : public IState_FSM
{
public:
void open(FSM* fsm) override { fsm->changeState("Open_without_CD", &FSM::open_tray); }
};
// Auto-register
static inline bool registered_ = []()
{
FSM::registerState<Closed_without_CD>("Closed_without_CD");
return true;
}();
//================================================================================ END
//---------------- File: main.cpp (Finite State Machine Pattern):
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
void print(std::string_view event) {std::cout << event << ":\n";}
int main()
try
{
FSM CD_Player("      CD Player", "Closed_without_CD");
print("open");          CD_Player.open();
print("insert_CD");     CD_Player.insert_CD(3);
print("close");         CD_Player.close();
print("play");          CD_Player.play();
print("next_song");     CD_Player.next_song();
print("play");          CD_Player.play();
print("previous_song"); CD_Player.previous_song();
print("pause");         CD_Player.pause();
print("play");          CD_Player.play();
print("next_song");     CD_Player.next_song();
print("stop");          CD_Player.stop();
print("open");          CD_Player.open();
print("play");          CD_Player.play();
print("open");          CD_Player.open();
print("remove_CD");     CD_Player.remove_CD();
print("close");         CD_Player.close();
print("insert_CD");     CD_Player.insert_CD(5);
}
catch(std::exception& e) {std::cout << "      Error: " << e.what() << '\n';}
// File: Open_with_CD.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class Open_with_CD : public IState_FSM
{
public:
void close(FSM* fsm) override { fsm->changeState("Closed_with_CD", &FSM::close_tray_and_dir); }
void remove_CD(FSM* fsm) override
{
fsm->nSongs = 0;
std::cout << fsm->name << ": A CD has been removed.\n";
fsm->CD_on_tray = false;
fsm->changeState("Open_without_CD");
}
void play(FSM* fsm) override { fsm->changeState("Playing", &FSM::close_tray_and_dir); }
};
// Auto-register
static inline bool registered_ = []()
{
FSM::registerState<Open_with_CD>("Open_with_CD");
return true;
}();
//================================================================================ END
// File: Open_without_CD.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class Open_without_CD : public IState_FSM
{
public:
void close(FSM* fsm) override { fsm->changeState("Closed_without_CD", &FSM::close_tray); }
void insert_CD(FSM* fsm, int n) override
{
fsm->nSongs = n;
std::cout << fsm->name << ": A CD has been inserted.\n";
fsm->CD_on_tray = true;
fsm->changeState("Open_with_CD");
}
};
// Auto-register
static inline bool registered_ = []()
{
FSM::registerState<Open_without_CD>("Open_without_CD");
return true;
}();
//================================================================================ END
// File: Paused.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class Paused : public IState_FSM
{
public:
void onEntry(FSM* fsm) override
{
std::cout << fsm->name << ": Turning ON cyan light.\n";
fsm->cian_light_on = true;
}
void onExit(FSM* fsm) override
{
std::cout << fsm->name << ": Turning OFF cyan light.\n";
fsm->cian_light_on = false;
}
void play(FSM* fsm) override 
{
fsm->changeState("Playing");
}
};
// Auto-register
static inline bool registered_ = []()
{
FSM::registerState<Paused>("Paused");
return true;
}();
//================================================================================ END
// File: Playing.cpp
// Author: Mario Galindo Queralt, Ph.D.
#include "FSM.h"
class Playing : public IState_FSM
{
public:
void onEntry(FSM* fsm) override
{
std::cout << fsm->name << ": Turning ON green light.\n";
fsm->green_light_on = true;
std::cout << fsm->name << ": Playing song number " << fsm->iSong << '\n';
fsm->playing = true;
}
void onExit(FSM* fsm) override
{
std::cout << fsm->name << ": Stop playing.\n";
fsm->playing = false;
std::cout << fsm->name << ": Turning OFF green light.\n";
fsm->green_light_on = false;
}
void open(FSM* fsm) override { fsm->changeState("Open_with_CD", &FSM::open_tray); }
void play(FSM* fsm) override { fsm->changeState("Playing", &FSM::reset); }
void stop(FSM* fsm) override { fsm->changeState("Closed_with_CD"); }
void next_song(FSM* fsm) override
{
fsm->iSong = (fsm->iSong < fsm->nSongs) ? fsm->iSong + 1 : 1;
std::cout << fsm->name << ": Playing song number " << fsm->iSong << '\n';
}
void previous_song(FSM* fsm) override
{
fsm->iSong = (fsm->iSong > 1) ? fsm->iSong - 1 : fsm->nSongs;
std::cout << fsm->name << ": Playing song number " << fsm->iSong << '\n';
}
void pause(FSM* fsm) override { fsm->changeState("Paused"); }
};
// Auto-register
static inline bool registered_ = []()
{
FSM::registerState<Playing>("Playing");
return true;
}();
//================================================================================ END
/**
* ============================================================================
* File: FSM.cpp (Modern Variant Implementation)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This file implements the FSM context logic and the main simulation loop.
* It uses std::visit to dispatch events to the current state stored in 
* the variant.
* ============================================================================
*/
#include "FSM.h"
void print(std::string_view event) { std::cout << "\n" << event << ":\n"; }
//--------------------------------------------------------- FSM Constructor:
FSM::FSM(std::string name) : name{std::move(name)}
{
std::get<Closed_without_CD>(state_).print_name();
}
//--------------------------------------------------------- FSM Dispatcher:
// We use [&] to capture any parameters (like 'n' in insert_CD) by reference.
#define DISPATCH(EVENT_CALL) \
std::visit([&](auto& s) { \
if constexpr (!std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) s.EVENT_CALL; \
}, state_);
void FSM::open()           { DISPATCH(open()); }
void FSM::close()          { DISPATCH(close()); }
void FSM::insert_CD(int n) { DISPATCH(insert_CD(n)); }
void FSM::remove_CD()      { DISPATCH(remove_CD()); }
void FSM::play()           { DISPATCH(play()); }
void FSM::stop()           { DISPATCH(stop()); }
void FSM::next_song()      { DISPATCH(next_song()); }
void FSM::previous_song()  { DISPATCH(previous_song()); }
void FSM::pause()          { DISPATCH(pause()); }
//--------------------------------------------------------- Main Simulation:
int main()
try
{
std::cout << "=== CD PLAYER (VARIANT STATE MACHINE) ===\n" << std::endl;
FSM CD_Player("    CD Player");
print("open");          CD_Player.open();
print("insert_CD");     CD_Player.insert_CD(3);
print("close");         CD_Player.close();
print("play");          CD_Player.play();
print("next_song");     CD_Player.next_song();
print("next_song");     CD_Player.next_song();
print("play");          CD_Player.play();
print("previous_song"); CD_Player.previous_song();
print("pause");         CD_Player.pause();
print("play");          CD_Player.play();
print("next_song");     CD_Player.next_song();
print("stop");          CD_Player.stop();
print("stop");          CD_Player.stop();
print("open");          CD_Player.open();
print("open");          CD_Player.open();
print("play");          CD_Player.play();
print("play");          CD_Player.play();
print("open");          CD_Player.open();
print("remove_CD");     CD_Player.remove_CD();
print("close");         CD_Player.close();
print("insert_CD");     CD_Player.insert_CD(5);
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
catch (char const* e) { std::cout << e; }
catch (const std::exception& e) { std::cout << e.what() << "\n"; }
//================================================================================ END
/**
* ============================================================================
* File: States.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This file contains the implementation of all state behaviors.
* 
* --- RAII (Resource Acquisition Is Initialization):
* Instead of explicit onEntry/onExit calls, we use constructors to turn 
* lights ON and destructors to turn them OFF. This guarantees that the 
* hardware state (lights) is always synchronized with the software state.
* ============================================================================
*/
#include "FSM.h"
#include <type_traits>
template <typename T>
struct fail : std::false_type {};
//--------------------------------------------------------- Closed_without_CD:
Closed_without_CD::Closed_without_CD(FSM* f) : fsm(f) { }
Closed_without_CD::~Closed_without_CD() { }
void Closed_without_CD::open()          { fsm->transitionTo<Open_without_CD>(&FSM::open_tray); }
void Closed_without_CD::close()         { }
void Closed_without_CD::insert_CD(int)  { throw ">>> The CD player has been destroyed inserting a CD by force!\n"; }
void Closed_without_CD::remove_CD()     { throw ">>> The CD player has been destroyed removing a CD by force!\n"; }
void Closed_without_CD::play()          { }
void Closed_without_CD::stop()          { }
void Closed_without_CD::next_song()     { }
void Closed_without_CD::previous_song() { }
void Closed_without_CD::pause()         { }
void Closed_without_CD::print_name()    { std::cout << "    Current state = Closed_without_CD\n"; }
//--------------------------------------------------------- Open_without_CD:
Open_without_CD::Open_without_CD(FSM* f) : fsm(f) { }
Open_without_CD::~Open_without_CD()    { }
void Open_without_CD::open()           { }
void Open_without_CD::close()          { fsm->transitionTo<Closed_without_CD>(&FSM::close_tray); }
void Open_without_CD::insert_CD(int n)
{
fsm->nSongs = n;
std::cout << fsm->name << ": A CD has been inserted.\n";
fsm->CD_on_tray = true;
fsm->transitionTo<Open_with_CD>();
}
void Open_without_CD::remove_CD()      { throw ">>> The CD player has been destroyed removing a CD by force!\n"; }
void Open_without_CD::play()           { }
void Open_without_CD::stop()           { }
void Open_without_CD::next_song()      { }
void Open_without_CD::previous_song()  { }
void Open_without_CD::pause()          { }
void Open_without_CD::print_name()     { std::cout << "    Current state = Open_without_CD\n"; }
//--------------------------------------------------------- Open_with_CD:
Open_with_CD::Open_with_CD(FSM* f) : fsm(f) { }
Open_with_CD::~Open_with_CD()      { }
void Open_with_CD::open()          { }
void Open_with_CD::close()         { fsm->transitionTo<Closed_with_CD>(&FSM::close_tray_and_dir); }
void Open_with_CD::insert_CD(int)  { throw ">>> The CD player has been destroyed inserting a CD by force!\n"; }
void Open_with_CD::remove_CD()
{
fsm->nSongs = 0;
std::cout << fsm->name << ": A CD has been removed.\n";
fsm->CD_on_tray = false;
fsm->transitionTo<Open_without_CD>();
}
void Open_with_CD::play()          { fsm->transitionTo<Playing>(&FSM::close_tray_and_dir); }
void Open_with_CD::stop()          { }
void Open_with_CD::next_song()     { }
void Open_with_CD::previous_song() { }
void Open_with_CD::pause()         { }
void Open_with_CD::print_name()    { std::cout << "    Current state = Open_with_CD\n"; }
//--------------------------------------------------------- Closed_with_CD:
Closed_with_CD::Closed_with_CD(FSM* f) : fsm(f)
{
std::cout << fsm->name << ": Turning ON yellow light.\n";
fsm->yellow_light_on = true;
}
Closed_with_CD::~Closed_with_CD()
{
std::cout << fsm->name << ": Turning OFF yellow light.\n";
fsm->yellow_light_on = false;
}
void Closed_with_CD::open()          { fsm->transitionTo<Open_with_CD>(&FSM::open_tray); }
void Closed_with_CD::close()         { }
void Closed_with_CD::insert_CD(int)  { throw ">>> The CD player has been destroyed inserting a CD by force!\n"; }
void Closed_with_CD::remove_CD()     { throw ">>> The CD player has been destroyed removing a CD by force!\n"; }
void Closed_with_CD::play()          { fsm->transitionTo<Playing>(&FSM::reset); }
void Closed_with_CD::stop()          { }
void Closed_with_CD::next_song()     { }
void Closed_with_CD::previous_song() { }
void Closed_with_CD::pause()         { }
void Closed_with_CD::print_name()    { std::cout << "    Current state = Closed_with_CD\n"; }
//--------------------------------------------------------- Playing:
Playing::Playing(FSM* f) : fsm(f)
{
std::cout << fsm->name << ": Turning ON green light.\n";
fsm->green_light_on = true;
std::cout << fsm->name << ": Playing song number " << fsm->iSong << '\n';
fsm->playing = true;
}
Playing::~Playing()
{
std::cout << fsm->name << ": Stop playing.\n";
fsm->playing = false;
std::cout << fsm->name << ": Turning OFF green light.\n";
fsm->green_light_on = false;
}
void Playing::open()         { fsm->transitionTo<Open_with_CD>(&FSM::open_tray); }
void Playing::close()        { }
void Playing::insert_CD(int) { throw ">>> The CD player has been destroyed inserting a CD by force!\n"; }
void Playing::remove_CD()    { throw ">>> The CD player has been destroyed removing a CD by force!\n"; }
void Playing::play()         { fsm->transitionTo<Playing>(&FSM::reset); }
void Playing::stop()         { fsm->transitionTo<Closed_with_CD>(); }
void Playing::next_song()
{
fsm->iSong = fsm->iSong < fsm->nSongs ? fsm->iSong + 1 : 1;
std::cout << fsm->name << ": Playing song number " << fsm->iSong << '\n';
}
void Playing::previous_song()
{
fsm->iSong = fsm->iSong > 1 ? fsm->iSong - 1 : fsm->nSongs;
std::cout << fsm->name << ": Playing song number " << fsm->iSong << '\n';
}
void Playing::pause()        { fsm->transitionTo<Paused>(); }
void Playing::print_name()   { std::cout << "    Current state = Playing\n"; }
//--------------------------------------------------------- Paused:
Paused::Paused(FSM* f) : fsm(f)
{
std::cout << fsm->name << ": Turning ON cian light.\n";
fsm->cian_light_on = true;
}
Paused::~Paused()
{
std::cout << fsm->name << ": Turning OFF cian light.\n";
fsm->cian_light_on = false;
}
void Paused::open()          { fsm->transitionTo<Open_with_CD>(&FSM::open_tray); }
void Paused::close()         { }
void Paused::insert_CD(int)  { throw ">>> The CD player has been destroyed inserting a CD by force!\n"; }
void Paused::remove_CD()     { throw ">>> The CD player has been destroyed removing a CD by force!\n"; }
void Paused::play()          { fsm->transitionTo<Playing>(); }
void Paused::stop()          { fsm->transitionTo<Closed_with_CD>(); }
void Paused::next_song()     { }
void Paused::previous_song() { }
void Paused::pause()         { }
void Paused::print_name()    { std::cout << "    Current state = Paused\n"; }
//================================================================================ END
/**
* ============================================================================
* File: Strategy.cpp (Simple Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Strategy pattern. It defines a family of 
* sorting algorithms, encapsulates each one, and makes them interchangeable 
* at runtime.
* 
* --- DESIGN PRINCIPLES:
* - Dependency Injection: The strategy is provided to the context, 
*                         decoupling the sorting algorithm from the data
*                         container.
* - Open/Closed Principle: New sorting algorithms can be added without 
*                          modifying the 'Sorter' class.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
//--------------------------------------------------------- Strategy Interface:
class SortStrategy
{
public:
virtual ~SortStrategy() = default;
virtual void sort(std::vector<int>& data) const = 0;
};
//--------------------------------------------------------- Concrete Strategy A:
class BubbleSort : public SortStrategy
{
public:
void sort(std::vector<int>& data) const override
{
std::cout << " [Strategy] Applying Bubble Sort\n";
for (size_t i = data.size(); i > 0; --i)
for (size_t j = 0; j < i - 1; ++j)
if (data[j] > data[j + 1])
std::swap(data[j], data[j + 1]);
}
};
//--------------------------------------------------------- Concrete Strategy B:
class ShellSort : public SortStrategy
{
public:
void sort(std::vector<int>& data) const override
{
std::cout << " [Strategy] Applying Shell Sort\n";
int n = static_cast<int>(data.size());
for (int g = n / 2; g > 0; g /= 2)
for (int i = g; i < n; ++i)
for (int j = i - g; j >= 0; j -= g)
if (data[j] > data[j + g])
std::swap(data[j], data[j + g]);
}
};
//--------------------------------------------------------- Context:
class Sorter
{
private:
std::unique_ptr<SortStrategy> strategy_;
public:
void setStrategy(std::unique_ptr<SortStrategy> strategy)
{
strategy_ = std::move(strategy);
}
void sortVector(std::vector<int>& data) const
{
if (strategy_) strategy_->sort(data);
else std::cout << " [System] No strategy set!\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== STRATEGY PATTERN (SIMPLE VERSION) ===\n" << std::endl;
std::vector<int> data1 = {30, 21, 6, 14, 8, 11, 10, 26, 12};
std::vector<int> data2 = data1;
Sorter sorter;
// 1. Use Bubble Sort
sorter.setStrategy(std::make_unique<BubbleSort>());
sorter.sortVector(data1);
std::cout << " Result: ";
for (int n : data1) std::cout << n << " ";
std::cout << "\n\n";
// 2. Use Shell Sort
sorter.setStrategy(std::make_unique<ShellSort>());
sorter.sortVector(data2);
std::cout << " Result: ";
for (int n : data2) std::cout << n << " ";
std::cout << "\n" << std::endl;
std::cout << "=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Strategy.cpp (Advanced Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation demonstrates "Strategy Composition". A single context 
* (SortAndPrint) delegates work to two different strategy families: 
* SortStrategy and PrintStrategy.
* 
* --- DEPENDENCY INJECTION:
* The client injects concrete strategies into the context at runtime, 
* allowing the sorting and printing behavior to vary independently without 
* modifying the core logic.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <algorithm>
//--------------------------------------------------------- Sort Strategies:
class SortStrategy
{
public:
virtual ~SortStrategy() = default;
virtual void sort(std::vector<int>& data) const = 0;
};
class BubbleSort : public SortStrategy
{
public:
void sort(std::vector<int>& data) const override
{
std::cout << " [Strategy] Applying Bubble Sort\n";
for (size_t i = data.size(); i > 0; --i)
for (size_t j = 0; j < i - 1; ++j)
if (data[j] > data[j + 1]) std::swap(data[j], data[j + 1]);
}
};
class ShellSort : public SortStrategy
{
public:
void sort(std::vector<int>& data) const override
{
std::cout << " [Strategy] Applying Shell Sort\n";
int n = static_cast<int>(data.size());
for (int g = n / 2; g > 0; g /= 2)
for (int i = g; i < n; ++i)
for (int j = i - g; j >= 0; j -= g)
if (data[j] > data[j + g]) std::swap(data[j], data[j + g]);
}
};
//--------------------------------------------------------- Print Strategies:
class PrintStrategy
{
public:
virtual ~PrintStrategy() = default;
virtual void print(const std::vector<int>& data) const = 0;
};
class FreePrint : public PrintStrategy
{
public:
void print(const std::vector<int>& data) const override
{
for (int n : data) std::cout << n << ' ';
std::cout << '\n';
}
};
class WidthPrint : public PrintStrategy
{
public:
void print(const std::vector<int>& data) const override
{
for (int n : data) std::cout << std::setw(2) << std::setfill('0') << n << ' ';
std::cout << '\n';
}
};
//--------------------------------------------------------- Context:
class SortAndPrint
{
private:
std::unique_ptr<SortStrategy>  sortImpl_;
// Abstract sort  strategy
std::unique_ptr<PrintStrategy> printImpl_;
// Abstract print strategy
public:
void setSort(std::unique_ptr<SortStrategy> sortImpl)
// Dependency Injection
{
sortImpl_ = std::move(sortImpl);
}
void setPrint(std::unique_ptr<PrintStrategy> printImpl)
// Dependency Injection
{
printImpl_ = std::move(printImpl);
}
void execute(std::vector<int>& data) const
{
if (sortImpl_) sortImpl_->sort(data);
if (printImpl_) printImpl_->print(data);
}
};
//--------------------------------------------------------- Main:
int main()
{
std::cout << "=== STRATEGY PATTERN (ADVANCED COMPOSITION) ===\n" << std::endl;
std::vector<int> data1 = {30, 21, 6, 14, 8, 11, 10, 26, 12};
std::vector<int> data2 = data1;
// Copy for the second test
SortAndPrint sorter;
// --- Sort with Bubble sort and FreePrint
std::cout << "Testing Bubble + FreePrint:\n";
sorter.setSort(std::make_unique<BubbleSort>());
sorter.setPrint(std::make_unique<FreePrint>());
sorter.execute(data1);
// --- Sort with Shell sort and WidthPrint
std::cout << "\nTesting Shell + WidthPrint:\n";
sorter.setSort(std::make_unique<ShellSort>());
sorter.setPrint(std::make_unique<WidthPrint>());
sorter.execute(data2);
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Static_Strategy.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation demonstrates the Static Strategy pattern using templates
* (Mixin Inheritance). Instead of virtual dispatch at runtime, the algorithm
* is bound at compile-time.
* 
* --- PERFORMANCE:
* Because there are no virtual functions, the compiler can inline the 
* sorting and printing logic directly into the 'execute' method, resulting
* in zero runtime overhead (Zero-Overhead Principle).
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
//--------------------------------------------------------- Sort Strategies:
struct BubbleSort
{
void sort(std::vector<int>& data) const
{
std::cout << " [Strategy] Applying Bubble Sort\n";
for (size_t i = data.size(); i > 0; --i)
for (size_t j = 0; j < i - 1; ++j)
if (data[j] > data[j + 1]) std::swap(data[j], data[j + 1]);
}
};
struct ShellSort
{
void sort(std::vector<int>& data) const
{
std::cout << " [Strategy] Applying Shell Sort\n";
int n = static_cast<int>(data.size());
for (int g = n / 2; g > 0; g /= 2)
for (int i = g; i < n; ++i)
for (int j = i - g; j >= 0; j -= g)
if (data[j] > data[j + g]) std::swap(data[j], data[j + g]);
}
};
//--------------------------------------------------------- Print Strategies:
struct FreePrint
{
void print(const std::vector<int>& data) const
{
for (int n : data) std::cout << n << ' ';
std::cout << '\n';
}
};
struct WidthPrint
{
void print(const std::vector<int>& data) const
{
for (int n : data)
std::cout << std::setw(2) << std::setfill('0') << n << ' ';
std::cout << '\n';
}
};
//--------------------------------------------------------- Context (Static):
template <typename SortAlgo, typename PrintAlgo>
class SortAndPrint : private SortAlgo, private PrintAlgo
{
public:
void execute(std::vector<int>& data) const
{
SortAlgo::sort(data);
PrintAlgo::print(data);
}
};
//--------------------------------------------------------- Main:
int main()
{
std::cout << "=== STRATEGY PATTERN (STATIC TEMPLATES) ===\n" << std::endl;
std::vector<int> data1 = {30, 21, 6, 14, 8, 11, 10, 26, 12};
std::vector<int> data2 = data1;
// 1. Compile-time composition: No pointers, no virtual functions.
SortAndPrint<BubbleSort, FreePrint> bubbleSorter;
std::cout << "Testing Bubble + FreePrint:\n";
bubbleSorter.execute(data1);
// 2. Another static composition
SortAndPrint<ShellSort, WidthPrint> shellSorter;
std::cout << "\nTesting Shell + WidthPrint:\n";
shellSorter.execute(data2);
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: TemplateMethod.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Template Method pattern. The base class 
* 'SortAndPrint' defines the algorithm skeleton 'sortAndPrintVector', 
* while concrete subclasses implement the specific 'sort' and 'print' steps.
* 
* --- HOLLYWOOD PRINCIPLE:
* "Don't call us, we will call you." The base class controls the flow 
* and calls the subclasses when needed.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <memory>
//--------------------------------------------------------- Base Template Class:
class SortAndPrint
{
private:
// Steps requiring peculiar implementations are private placeholders.
// Derived classes override these to provide the "how".
virtual void sort(std::vector<int>& data) = 0;
virtual void print(const std::vector<int>& data) const = 0;
public:
virtual ~SortAndPrint() = default;
/**
* THE TEMPLATE METHOD (Algorithm Skeleton)
* This method implements the Non-Virtual Interface (NVI) idiom.
* It defines a stable sequence of steps that subclasses cannot bypass.
*/
void sortAndPrintVector(std::vector<int>& data)
{
// 1. [INVARIANT CODE] Pre-processing:
// Perform mandatory administrative tasks such as input validation, 
// logging the start of the operation, or starting performance timers.
sort(data);
// Call to the specific "how" (variant part)
// 2. [INVARIANT CODE] Intermediate Logic:
// Execute global consistency checks or verify invariants to ensure 
// the algorithm step (sort) left the system in a valid state.
print(data);
// Call to the second specific step (variant part)
// 3. [INVARIANT CODE] Post-processing:
// Finalize administrative actions such as measuring elapsed time, 
// reporting telemetry metrics, or releasing global resources.
}
};
//--------------------------------------------------------- Concrete Classes:
class BubbleSort : public SortAndPrint
{
private:
void sort(std::vector<int>& data) override
{
for (size_t i = data.size(); i > 0; --i)
for (size_t j = 0; j < i - 1; ++j)
if (data[j] > data[j + 1]) std::swap(data[j], data[j + 1]);
}
void print(const std::vector<int>& data) const override
{
std::cout << "Bubble: ";
for (int n : data) std::cout << n << ' ';
std::cout << '\n';
}
};
class ShellSort : public SortAndPrint
{
private:
void sort(std::vector<int>& data) override
{
int n = static_cast<int>(data.size());
for (int g = n / 2; g > 0; g /= 2)
for (int i = g; i < n; ++i)
for (int j = i - g; j >= 0; j -= g)
if (data[j] > data[j + g]) std::swap(data[j], data[j + g]);
}
void print(const std::vector<int>& data) const override
{
std::cout << "Shell:  ";
for (int n : data) std::cout << std::setw(2) << std::setfill('0') << n << ' ';
std::cout << '\n';
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== TEMPLATE METHOD PATTERN ===\n" << std::endl;
std::vector<int> data1 = {30, 21, 6, 14, 8, 11, 10, 26, 12, 1, 15, 20, 8, 11, 5, 26, 30, 3};
std::vector<int> data2 = data1;
std::cout << "Initial Vector: ";
for (int n : data1) std::cout << n << ' ';
std::cout << "\n\n";
// We use unique_ptr for safety
std::unique_ptr<SortAndPrint> sorter;
// 1. Sort with Bubble Sort
sorter = std::make_unique<BubbleSort>();
sorter->sortAndPrintVector(data1);
// 2. Sort with Shell Sort
sorter = std::make_unique<ShellSort>();
sorter->sortAndPrintVector(data2);
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Visitor.cpp (Simple Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program solves the "Double Dispatch" problem using the Visitable/Visitor 
* logic. In C++, virtual functions only resolve the type of the caller 
* at runtime. To resolve the types of BOTH objects (e.g., in a collision), 
* we need a second dispatch.
* 
* Logic:
* 1. first->intersect(*second)
// Runtime call on 'first'
* 2. second->intersect(*this)
// Runtime call on 'second' with 'this'
*
// as a concrete type.
* 
* --- NOTE ON DUAL ROLE:
* In this particular double dispatch scenario, every object acts as both 
* the 'Visitable' (the receiver) and the 'Visitor' (the executor). To reflect 
* this dual responsibility, the base interface is named 'Visitable_Visitor'.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <memory>
class Circle;
class Triangle;
//------------------------------------------------- Base Interface (Dual Role):
class Visitable_Visitor
{
public:
virtual ~Visitable_Visitor() = default;
// The entry point for the double dispatch
virtual void intersect(const Visitable_Visitor& other) const = 0;
// The overloaded verification methods for concrete types
virtual void intersect(const Circle& other) const = 0;
virtual void intersect(const Triangle& other) const = 0;
};
//--------------------------------------------------------- Concrete Shape A:
class Circle : public Visitable_Visitor
{
public:
void intersect(const Visitable_Visitor& other) const override 
{
// First dispatch: 'other' is polymorphic, but '*this' is known as Circle.
other.intersect(*this); 
}
void intersect(const Circle&) const override 
{
std::cout << " [Verification] Checking intersection: Circle <-> Circle\n";
}
void intersect(const Triangle&) const override 
{
std::cout << " [Verification] Checking intersection: Circle <-> Triangle\n";
}
};
//--------------------------------------------------------- Concrete Shape B:
class Triangle : public Visitable_Visitor
{
public:
void intersect(const Visitable_Visitor& other) const override 
{
// First dispatch: 'other' is polymorphic, but '*this' is known as Triangle.
other.intersect(*this); 
}
void intersect(const Circle&) const override 
{
std::cout << " [Verification] Checking intersection: Triangle <-> Circle\n";
}
void intersect(const Triangle&) const override 
{
std::cout << " [Verification] Checking intersection: Triangle <-> Triangle\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== DOUBLE DISPATCH (VISITABLE_VISITOR SIMULATION) ===\n" << std::endl;
// Setup a heterogeneous collection of shapes using the dual-role interface
std::vector<std::unique_ptr<Visitable_Visitor>> shapes;
shapes.push_back(std::make_unique<Triangle>());
shapes.push_back(std::make_unique<Circle>());
// Test every possible combination using nested loops.
// The output will verify that the specific types were correctly identified.
for (const auto& s1 : shapes)
{
for (const auto& s2 : shapes)
{
s1->intersect(*s2);
}
}
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Visitor_GoF.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* Visitor pattern implemented to solve the "double dispatch problem".
* This implementation keeps the object structure (Visitable) decoupled from
* the operations (Visitor).
* 
* --- RECOMPILATION ANALYSIS:
* 1. Incrementing a new Visitor -> Neither existing Visitors nor Visitable 
*    classes need to be recompiled.
* 2. Incrementing a new Visitable -> Visitable classes do not need to be 
*    recompiled, BUT ALL existing Visitor classes MUST be recompiled!
* ============================================================================
*/
#include "Visitor.h"
#include <vector>
#include <iostream>
#include <memory>
//--------------------------------------------------------- Family Forward Decls:
class Red;
class Blue;
//--------------------------------------------------------- Visitor Interface:
// Needs to know about Red and Blue, so it is defined here instead of the 
// generic header to prevent cyclic dependencies in the infrastructure.
class Visitor
{
protected:
Visitor() = default;
public:
virtual ~Visitor() = default;
// Visitor depends upon Visitable types. 
// Default implementation throws an error to ensure safety.
virtual void visit(Red&)  { throw dispatch_error(typeid(*this).name(), "Red"); } 
virtual void visit(Blue&) { throw dispatch_error(typeid(*this).name(), "Blue"); } 
};
//--------------------------------------------------------- Dispatch Actions:
// These represent the exclusive logic for each specific combination.
class Triangle;
class Circle;
void redTriangle(Red&, Triangle&)   { std::cout << " -> Action: Exclusive Red Triangle function\n"; }
void redCircle(Red&, Circle&)       { std::cout << " -> Action: Exclusive Red Circle function\n"; }
void blueTriangle(Blue&, Triangle&) { std::cout << " -> Action: Exclusive Blue Triangle function\n"; }
void blueCircle(Blue&, Circle&)     { std::cout << " -> Action: Exclusive Blue Circle function\n"; }
//--------------------------------------------------------- Colors (Visitable):
class Red : public Visitable
{
public:
// Every Visitable must include this line to perform the handshake
void accept(Visitor& visitor) override { visitor.visit(*this); } 
};
class Blue : public Visitable
{
public:
void accept(Visitor& visitor) override { visitor.visit(*this); }
};
//--------------------------------------------------------- Shapes (Visitors):
class Triangle : public Visitor
{
public:
// Overrides are defined selectively or for all types
void visit(Red& red)   override { redTriangle(red, *this); } 
void visit(Blue& blue) override { blueTriangle(blue, *this); }
};
class Circle : public Visitor
{
public:
void visit(Red& red) override { redCircle(red, *this); }
// Note: Blue is intentionally NOT overridden to test the exception logic.
// void visit(Blue& blue) override { blueCircle(blue, *this); }
};
//--------------------------------------------------------- Simulation Engine:
using Visitable_ptr = std::unique_ptr<Visitable>;
using Visitor_ptr = std::unique_ptr<Visitor>;
/**
* dispatch_all_combinations:
* Iterates through the collections and executes the double dispatch.
* It demonstrates how the symmetry of double_dispatch() makes the 
* call natural regardless of the order of objects.
*/
void dispatch_all_combinations(
std::vector<Visitable_ptr>& colors, 
std::vector<Visitor_ptr>& shapes)
{
for (auto& visitable : colors)
{
for (auto& visitor : shapes)
{
try
{
// Handshake starts here!
double_dispatch(*visitable, *visitor); 
}
catch (const dispatch_error& e)
{
std::cerr << " [System] " << e.what() << std::endl;
}
}
}
}
//--------------------------------------------------------- Main Entry Point:
int main()
{
std::cout << "=== VISITOR PATTERN (GOF CLASSIC SIMULATION) ===\n" << std::endl;
// 1. Setup the Visitable collection (The "Colors")
std::vector<Visitable_ptr> colors;
colors.push_back(std::make_unique<Red>());
colors.push_back(std::make_unique<Blue>());
// 2. Setup the Visitor collection (The "Shapes")
std::vector<Visitor_ptr> shapes;
shapes.push_back(std::make_unique<Triangle>());
shapes.push_back(std::make_unique<Circle>());
// 3. Process the matrix of interactions
dispatch_all_combinations(colors, shapes);
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Visitor_RTTI.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation uses Runtime Type Information (RTTI) to allow creating 
* new Visitable or Visitor classes without recompiling existing ones.
* 
* It solves the "Cyclic Dependency" of the GoF pattern by using dynamic_cast 
* to verify interface compatibility at runtime.
* ============================================================================
*/
#include "Visitor_RTTI.h"
#include <vector>
#include <iostream>
#include <memory>
//--------------------------------------------------------- Forward Decls:
class Red;
class Blue;
class Triangle;
class Circle;
//--------------------------------------------------------- Dispatch Actions:
void redTriangle(Red&, Triangle&)   { std::cout << " -> Exclusive RED + TRIANGLE function\n"; }
void redCircle(Red&, Circle&)       { std::cout << " -> Exclusive RED + CIRCLE function\n"; }
void blueTriangle(Blue&, Triangle&) { std::cout << " -> Exclusive BLUE + TRIANGLE function\n"; }
void blueCircle(Blue&, Circle&)     { std::cout << " -> Exclusive BLUE + CIRCLE function\n"; }
//--------------------------------------------------------- Colors (Visitable):
// Each class must inherit from 'Visitable' and the 'As<T>' helper.
class Red : public Visitable, public As<Red>
{
};
class Blue : public Visitable, public As<Blue>
{
};
//--------------------------------------------------------- Shapes (Visitor):
// Visitors inherit from 'Visitor' and any number of 'Visit<T>' interfaces.
class Triangle : public Visitor, 
public Visit<Red>, 
public Visit<Blue>
{
public:
void visit(Red& red)   override { redTriangle(red, *this); }
void visit(Blue& blue) override { blueTriangle(blue, *this); }
};
class Circle : public Visitor, 
public Visit<Red>
{
public:
void visit(Red& red) override { redCircle(red, *this); }
// Blue is intentionally NOT implemented here to test Dispatch_error
};
//--------------------------------------------------------- Simulation Engine:
using Visitable_ptr = std::unique_ptr<Visitable>;
using Visitor_ptr = std::unique_ptr<Visitor>;
void dispatchAllCombinations(
std::vector<Visitable_ptr>& colors, 
std::vector<Visitor_ptr>& shapes)
{
for (auto& visitable : colors)
{
for (auto& visitor : shapes)
{
try
{
double_dispatch(*visitable, *visitor);
}
catch (const Dispatch_error& e)
{
std::cerr << " [System] " << e.what() << std::endl;
}
}
}
}
//--------------------------------------------------------- Main Entry Point:
int main()
{
std::cout << "=== VISITOR PATTERN (RTTI / ACYCLIC VERSION) ===\n" << std::endl;
std::vector<Visitable_ptr> colors;
colors.push_back(std::make_unique<Red>());
colors.push_back(std::make_unique<Blue>());
std::vector<Visitor_ptr> shapes;
shapes.push_back(std::make_unique<Triangle>());
shapes.push_back(std::make_unique<Circle>());
dispatchAllCombinations(colors, shapes);
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Visitor_Modern.cpp (Optimized Variant Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Visitor pattern using std::variant.
* We optimize the "Double Dispatch" matrix to avoid redundant calculations:
* 1. Intersection(A, B) is treated as the same as Intersection(B, A).
* 2. We skip self-intersection (A, A) if not desired.
* 3. We use nested loops: for (i=0; i<N; ++i) for (j=i+1; j<N; ++j).
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <variant>
#include <algorithm>
//--------------------------------------------------------- Plain Data Structs:
struct Circle   { int id_; explicit Circle(int id) : id_{id} {} };
struct Triangle { int id_; explicit Triangle(int id) : id_{id} {} };
using Shape = std::variant<Circle, Triangle>;
//--------------------------------------------------------- Interaction Logic:
struct CollisionEngine
{
// Helper to perform the actual math once and reuse it
void intersect(int id1, int id2, const std::string& type) const
{
std::cout << " [System] Collision detected between " << type 
<< " [" << id1 << "] and [" << id2 << "]\n";
}
// Implementation for Circle <-> Circle
void operator()(const Circle& a, const Circle& b) const
{
if (a.id_ != b.id_) intersect(a.id_, b.id_, "Circle-Circle");
}
// Implementation for Triangle <-> Triangle
void operator()(const Triangle& a, const Triangle& b) const
{
if (a.id_ != b.id_) intersect(a.id_, b.id_, "Triangle-Triangle");
}
// Implementation for Circle <-> Triangle
void operator()(const Circle& a, const Triangle& b) const
{
intersect(a.id_, b.id_, "Circle-Triangle");
}
// Implementation for Triangle <-> Circle (Reverse order)
// We simply redirect to the Circle-Triangle logic to maintain OCP
void operator()(const Triangle& a, const Circle& b) const
{
(*this)(b, a);
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== MODERN VARIANT DISPATCH (OPTIMIZED) ===\n" << std::endl;
std::vector<Shape> shapes;
shapes.emplace_back(Circle{1});
shapes.emplace_back(Triangle{2});
shapes.emplace_back(Circle{3});
shapes.emplace_back(Triangle{4});
// Optimized O(N(N-1)/2) loop
for (size_t i = 0; i < shapes.size(); ++i)
{
for (size_t j = i + 1; j < shapes.size(); ++j)
{
std::visit(CollisionEngine{}, shapes[i], shapes[j]);
}
}
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Visitor_Traditional_GoF.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation demonstrates the classic GoF Visitor pattern.
* We have a Document structure (Paragraphs, Images, Hyperlinks) and we
* perform different operations on it (HTML Export, Text Extraction) without
* modifying the element classes themselves.
* ============================================================================
*/
#include "Visitor.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
//--------------------------------------------------------- Concrete Visitables:
class Paragraph : public Visitable
{
private:
std::string text_;
public:
explicit Paragraph(std::string text) : text_{std::move(text)} {}
void accept(Visitor& v) override { v.visit(*this); }
std::string getText() const { return text_; }
};
class Image : public Visitable
{
private:
std::string url_;
public:
explicit Image(std::string url) : url_{std::move(url)} {}
void accept(Visitor& v) override { v.visit(*this); }
std::string getUrl() const { return url_; }
};
class Hyperlink : public Visitable
{
private:
std::string url_;
std::string label_;
public:
Hyperlink(std::string url, std::string label) 
: url_{std::move(url)}, label_{std::move(label)} {}
void accept(Visitor& v) override { v.visit(*this); }
std::string getUrl() const { return url_; }
std::string getLabel() const { return label_; }
};
//--------------------------------------------------------- Concrete Visitors:
class HtmlExport : public Visitor
{
public:
void visit(Paragraph& p) override
{
std::cout << "<p>" << p.getText() << "</p>\n";
}
void visit(Image& i) override
{
std::cout << "<img src=\"" << i.getUrl() << "\" />\n";
}
void visit(Hyperlink& h) override
{
std::cout << "<a href=\"" << h.getUrl() << "\">" << h.getLabel() << "</a>\n";
}
};
class TextExtract : public Visitor
{
public:
void visit(Paragraph& p) override
{
std::cout << "Text: " << p.getText() << "\n";
}
void visit(Image& i) override
{
std::cout << "Image found at: " << i.getUrl() << "\n";
}
void visit(Hyperlink& h) override
{
std::cout << "Link: " << h.getLabel() << " [" << h.getUrl() << "]\n";
}
};
//--------------------------------------------------------- Main:
int main()
{
std::cout << "=== VISITOR PATTERN (TRADITIONAL GOF) ===\n" << std::endl;
std::vector<std::unique_ptr<Visitable>> document;
document.push_back(std::make_unique<Paragraph>("Hello World!"));
document.push_back(std::make_unique<Image>("logo.png"));
document.push_back(std::make_unique<Hyperlink>("https:
//cpp.org", "C++ Site"));
HtmlExport htmlExporter;
TextExtract textExtractor;
std::cout << "--- HTML Export:\n";
for (auto& element : document)
{
element->accept(htmlExporter);
}
std::cout << "\n--- Text Extraction:\n";
for (auto& element : document)
{
element->accept(textExtractor);
}
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Visitor_Modern.cpp (std::variant Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This implementation uses C++17 'std::variant' and 'std::visit' to replace 
* traditional virtual inheritance with type-safe unions.
* 
* THE MODERN ADVANTAGE:
* 1. Non-Intrusive: The element classes (Paragraph, etc.) don't need 'accept()'.
* 2. Static Safety: 'std::visit' forces you to handle all types in the variant.
* 3. Zero Inheritance: We don't need a base Visitable class, reducing 
*    the complexity of the object structure.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <vector>
#include <variant>
//--------------------------------------------------------- Plain Data Structs:
// No inheritance, no virtual methods. Pure data elements.
struct Paragraph
{
std::string text_;
explicit Paragraph(std::string text) : text_{std::move(text)} {}
};
struct Image
{
std::string url_;
explicit Image(std::string url) : url_{std::move(url)} {}
};
struct Hyperlink
{
std::string url_;
std::string label_;
Hyperlink(std::string url, std::string label) 
: url_{std::move(url)}, label_{std::move(label)} {}
};
// Define the DocumentElement as a variant of all possible types
using DocumentElement = std::variant<Paragraph, Image, Hyperlink>;
//--------------------------------------------------------- Modern Visitors:
// A Visitor in C++17 is simply a struct/class with overloaded operator()
struct HtmlExporter
{
void operator()(const Paragraph& p) const { std::cout << "<p>" << p.text_ << "</p>\n"; }
void operator()(const Image& i)     const { std::cout << "<img src=\"" << i.url_ << "\" />\n"; }
void operator()(const Hyperlink& h) const { std::cout << "<a href=\"" << h.url_ << "\">" << h.label_ << "</a>\n"; }
};
struct TextExtractor
{
void operator()(const Paragraph& p) const { std::cout << "Text: " << p.text_ << "\n"; }
void operator()(const Image& i)     const { std::cout << "Image found at: " << i.url_ << "\n"; }
void operator()(const Hyperlink& h) const { std::cout << "Link: " << h.label_ << " [" << h.url_ << "]\n"; }
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== VISITOR PATTERN (MODERN VARIANT) ===\n" << std::endl;
// The document is now a vector of variants
std::vector<DocumentElement> document;
document.push_back(Paragraph("Hello World!"));
document.push_back(Image("logo.png"));
document.push_back(Hyperlink("https:
//cpp.org", "C++ Site"));
// 1. HTML Export
std::cout << "--- HTML Export:\n";
for (const auto& element : document)
{
std::visit(HtmlExporter{}, element);
}
// 2. Text Extraction
std::cout << "\n--- Text Extraction:\n";
for (const auto& element : document)
{
std::visit(TextExtractor{}, element);
}
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Framework.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This is the core engine of the Framework. It implements the "Hollywood 
* Principle" (Don't call us, we call you) by controlling the application 
* lifecycle.
* 
* --- EXECUTION FLOW:
* The framework executes the algorithm steps defined in 'App.h' in a 
* fixed order: start -> read -> compute -> write -> result -> final.
* Clients only provide the concrete implementation of these steps.
* ============================================================================
*/
#include "App.h"
#include <memory>
int main()
{
// Here is created your own App
// We use unique_ptr to manage the App lifetime automatically
std::unique_ptr<App> app{createApp()};
// (1) Do initial actions
app->start();
// (2) Read from input
while (app->read())
{
// (3) Perform some computations
app->compute();
// (4) Write some output
app->write();
}
// (5) Produce a final result
app->result();
// (6) Do final actions
app->final();
}
//========================================================================= END
/**
* ============================================================================
* File: up.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This is a concrete implementation of the 'App' framework.
* It reads a text file line by line and converts every character to 
* uppercase, demonstrating how a specific client fills the framework 
* skeleton with concrete functionality.
* ============================================================================
*/
#include "../Framework/App.h"
#include <iostream>
#include <fstream>
#include <cctype>
// std::toupper
#include <string>
class up : public App 
{
private:
char c_{0};
std::ifstream fs_;
public:
void start() override
// (1)
{
std::string filename;
std::getline(std::cin, filename);
fs_.open(filename);
if (!fs_)
{
std::cerr << "Can't open " << filename << '\n';
std::exit(1);
}
}
bool read() override
// (2)
{
c_ = static_cast<char>(fs_.get());
// Read a character
return fs_.good();
// Return EOF condition
}
void write() override
// (4)
{
std::cout << static_cast<char>(std::toupper(c_));
}
void final() override
// (6)
{
fs_.close();
}
};
// Factory method to create the concrete application instance
App* createApp() 
{
return new up;
}
//================================================================================ END
/**
* ============================================================================
* File: wc.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program implements a word and character count utility using the 'App'
* framework. It showcases how different applications can reuse the same 
* execution skeleton ('Framework/Framework.cpp') while providing unique 
* processing logic.
* ============================================================================
*/
#include "../Framework/App.h"
#include <iostream>
// Define your new customized application
class wc : public App 
{
private:
char c_;
int nChar_{0}, nWord_{1}, nLine_{1};
public:
bool read() override
// (2)
{
std::cin.get(c_);
// Read a character
return std::cin.good();
// Return EOF cond.
}
void compute() override
// (3)
{
++nChar_;
if (c_ == ' ' || c_ == '\n') ++nWord_;
if (c_ == '\n') ++nLine_;
}
void result() override
// (5)
{
std::cout << "lines=" << nLine_ << ' ';
std::cout << "words=" << nWord_ << ' ';
std::cout << "chars=" << nChar_ << '\n';
}
};
App* createApp() 
{
return new wc;
}
//========================================================================= END
/**
* ============================================================================
* File: NullObject.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Null Object pattern. We use a 'NullLogger' 
* as a safe substitute for a real logger. 
* 
* --- ARCHITECTURAL NOTE:
* By implementing 'NullLogger' as a Singleton, we ensure that we don't 
* waste memory creating multiple "do-nothing" objects. The client code 
* treats the 'NullLogger' exactly like a real 'Logger', eliminating the 
* need for explicit 'if (logger != nullptr)' checks.
* ============================================================================
*/
#include <iostream>
#include <string>
#include <memory>
#include <vector>
//--------------------------------------------------------- Logger Interface:
class Logger
{
public:
virtual ~Logger() = default;
virtual void log(const std::string& message) const = 0;
};
//--------------------------------------------------------- Real Logger:
class ConsoleLogger : public Logger
{
private:
std::string name_;
public:
explicit ConsoleLogger(std::string name) : name_{std::move(name)} { }
void log(const std::string& message) const override
{
std::cout << " [" << name_ << "] " << message << "\n";
}
};
//--------------------------------------------------------- Null Logger:
// The Null Object provides "do-nothing" behavior.
// We implement it as a Singleton since it has no state.
class NullLogger : public Logger
{
private:
NullLogger() = default;
public:
static NullLogger& getInstance()
{
static NullLogger instance;
return instance;
}
// No-op implementation: does nothing.
void log(const std::string&) const override { }
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== NULL OBJECT PATTERN ===\n" << std::endl;
// Two different real loggers
ConsoleLogger logger1{"Console-A"};
ConsoleLogger logger2{"Console-B"};
// The shared Null Object (Singleton)
NullLogger& null = NullLogger::getInstance();
// Only one instance
std::vector<Logger*> loggers;
loggers.push_back(&logger1);
loggers.push_back(&null);
loggers.push_back(&logger2);
loggers.push_back(&null);
std::cout << "Broadcasting logs to all loggers (both real and null):\n";
for (const auto* logger : loggers) logger->log("System update in progress.");
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: NullObject.cpp (Modern Variant Version)
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the Null Object pattern using std::variant 
* and std::monostate. 
* 
* --- THE MODERN APPROACH:
* Instead of creating a 'NullLogger' class via inheritance, we use 
* 'std::monostate' as a type-safe alternative to represent the absence 
* of a logger. 
* 
* --- THE ADVANTAGE:
* The client code handles the logging through 'std::visit'. If the variant 
* contains 'std::monostate', the "Null" behavior is executed (doing nothing). 
* This eliminates raw pointers and prevents runtime crashes while maintaining
* value semantics and stack allocation.
* ============================================================================
*/
#include <iostream>
#include <variant>
#include <vector>
#include <string>
//--------------------------------------------------------- Real Logger:
// A simple struct that performs actual work. No inheritance required.
struct ConsoleLogger
{
std::string name;
void log(const std::string& message) const
{
std::cout << " [" << name << "] " << message << "\n";
}
};
//--------------------------------------------------------- The Nullable Type:
// Logger can be either a real ConsoleLogger or "Nothing" (std::monostate)
// std::monostate acts as the Null Object here.
using Logger = std::variant<std::monostate, ConsoleLogger>;
//--------------------------------------------------------- The Dispatcher:
// This Visitor handles both the real logger and the Null Object case.
struct LogVisitor
{
std::string message;
// 1. Logic for the Real Object
void operator()(const ConsoleLogger& logger) const
{
logger.log(message);
}
// 2. Logic for the Null Object (std::monostate)
// This is the "do-nothing" behavior encapsulated as a type.
void operator()(std::monostate) const
{
// Intentionally empty: no-op behavior
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== NULL OBJECT PATTERN (VARIANT VERSION) ===\n" << std::endl;
// A collection of loggers using value semantics (no pointers!)
std::vector<Logger> loggers;
loggers.emplace_back(ConsoleLogger{"Console-A"});
// Real Object
loggers.emplace_back(std::monostate{});
// The Null Object
loggers.emplace_back(ConsoleLogger{"Console-B"});
// Real Object
loggers.emplace_back(std::monostate{});
// The Null Object
std::cout << "Broadcasting logs to all loggers (including the monostate Null Object):\n";
// The compiler ensures both cases (Real and Null) are handled via std::visit
for (const auto& logger : loggers)
std::visit(LogVisitor{"System update in progress."}, logger);
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Variant.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* Demonstrates the use of std::variant and std::visit (C++17).
* Includes a custom 'Logger' class implementing "The Rule of Seven" to
* trace memory lifecycle and ownership changes within the variant.
* 
* --- MANUAL ACCESS:
* Also demonstrates how to safely access the content of a variant without 
* a visitor using std::get_if, providing a pointer-based check.
* ============================================================================
*/
// Developed by MGQ at https://wandbox.org/permlink/6d5UVbWrezYijARP
#include <iostream>
#include <variant>
#include <stack>
#include <string>
#include <utility>
//---------------------------------------------------------------------------------------- Logger:
// As an example, define a logger class to follow it:
struct Logger
{
std::string s;
// Define the Rule of seven:
Logger()
{
std::cout << "----- Logger 1 DC: Default Constructor\n";
}
Logger(Logger const& other) : s{other.s}
{
std::cout << "----- Logger 2 CC: Copy Constructor: " << s << '\n';
}
Logger(Logger&& other) noexcept : s{std::move(other.s)}
{
std::cout << "----- Logger 3 MC: Move Constructor: " << s << '\n';
}
Logger& operator=(Logger const& other)
{
std::cout << "----- Logger 4 CA: Copy Assignment: " << other.s << '\n';
s = other.s;
return *this;
}
Logger& operator=(Logger&& other) noexcept
{
std::cout << "----- Logger 5 MA: Move Assignment: " << other.s << '\n';
s = std::move(other.s);
return *this;
}
~Logger()
{
std::cout << "----- Logger 6 De: Destructor: " << s << '\n';
}
explicit Logger(std::string s) : s{std::move(s)}
{
std::cout << "----- Logger 7 PC: Parametric Constructor: " << this->s << '\n';
}
};
std::ostream& operator << (std::ostream& os, Logger const& g)
{
os << g.s;
return os;
}
//------------------------------------------------------------------------------------------ Main:
int main()
{
//----------------------------------------------------- Declare the variant (union):
using var_t = std::variant<int, double, char, Logger, const char*, std::string>;
//----------------------------------------------- Write the sizes of the components:
std::cout << "\n----------------------- Size of components:\n";
std::cout << "size of int:         " << sizeof(int) << '\n';
std::cout << "size of double:      " << sizeof(double) << '\n';
std::cout << "size of char:        " << sizeof(char) << '\n';
std::cout << "size of Logger:      " << sizeof(Logger) << '\n';
std::cout << "size of const char*: " << sizeof(const char*) << '\n';
std::cout << "size of std::string: " << sizeof(std::string) << '\n';
std::cout << "\nsize of var_t:       " << sizeof(var_t) << '\n';
//------------------ Create a visitor (define a operator() to work with each type):
struct Visitor
{
int in{0}, db{0}, ch{0}, lo{0}, cp{0}, st{0};
void operator()(int i)                { std::cout << ">> int: "         << i << '\n'; ++in; }
void operator()(double d)             { std::cout << ">> double: "      << d << '\n'; ++db; }
void operator()(char c)               { std::cout << ">> char: "        << c << '\n'; ++ch; }
void operator()(const Logger& l)      { std::cout << ">> Logger: "      << l << '\n'; ++lo; }
void operator()(const char* p)        { std::cout << ">> const char*: " << p << '\n'; ++cp; }
void operator()(const std::string& s) { std::cout << ">> std::string: " << s << '\n'; ++st; }
} visitor;
//----------------------------------------------------- Create a stack of variants:
std::stack<var_t> myStack;
//--------------------------------------------- Operate with the stack of variants:
std::cout << "\n--------------------------- Fill the stack:\n";
using namespace std::string_literals;
myStack.emplace(12);
myStack.emplace(8.3);
myStack.emplace('c');
myStack.emplace(9);
myStack.emplace(Logger{"I am a Logger"});
myStack.emplace("C string 1");
myStack.emplace("I am std::string"s);
myStack.emplace("C string 2");
std::cout << "\n--------------------------- Read the stack:\n";
while (myStack.size() > 0)
{
var_t v{std::move(myStack.top())};
myStack.pop();
//---- Visit the variant:
std::visit(visitor, v);
}
std::cout << "\n---------------------------------- Results:\n";
std::cout << "Num int:         " << visitor.in << '\n';
std::cout << "Num double:      " << visitor.db << '\n';
std::cout << "Num char:        " << visitor.ch << '\n';
std::cout << "Num Logger:      " << visitor.lo << '\n';
std::cout << "Num const char*: " << visitor.cp << '\n';
std::cout << "Num string:      " << visitor.st << '\n';
//------------------------------------------------- Manual access without Visitor:
std::cout << "\n--------------------------- Manual Access (std::get_if):\n";
var_t manualVar = 3.1415;
// get_if returns a pointer if the type matches, or nullptr otherwise
if (auto* dVal = std::get_if<double>(&manualVar))
std::cout << " Found double value: " << *dVal << "\n";
if ([[maybe_unused]] auto* iVal = std::get_if<int>(&manualVar))
std::cout << " This won't print, since manualVar is not an int.\n";
else
std::cout << " Confirmed: manualVar does not contain an int.\n";
std::cout << "\n--------------------------------------- END\n";
}
//============================================================================================ END
/**
* ============================================================================
* File: VariantValueClone.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the "Value Semantics" approach using std::variant
* as a modern alternative to traditional polymorphism and the Prototype pattern.
* 
* --- THE ARCHITECTURAL SHIFT:
* In the example (32/CRTP/03_PolymorphicClone), we used heap allocation and 
* a CRTP Mixin to handle cloning. Here, we store objects directly on the stack 
* (inside the variant). 
* 
* --- ADVANTAGES:
* 1. Automatic Deep Copy: Since std::variant and the concrete classes (Square, 
*    Circle) have value semantics, cloning the entire container is handled 
*    automatically by the language.
* 2. No Pointers: We eliminate memory management concerns and the need for 
*    smart pointers or explicit 'clone()' methods.
* 3. Static Dispatch: We use std::visit to invoke behaviors like 'draw()', 
*    allowing the compiler to optimize the calls without relying on vtables.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <string>
#include <variant>
//---------------------------------------------------- Abstract Base Shape:
class Shape
// Base
{
public:
virtual ~Shape() = default;
virtual void draw() const = 0;
};
//-------------------------------------------------- Concrete Shape Square:
class Square : public Shape
{
public:
Square()
// Constructor
{
std::cout << " [Constructor] Created a new Square\n";
}
Square(const Square& other) : Shape(other)
// Copy constructor
{
std::cout << " [Copy Constructor] Copied a Square\n";
}
Square(Square&& other) noexcept : Shape(std::move(other))
{
std::cout << " [Move] Moved a Square\n";
// Move constructor
}
void draw() const override
{
std::cout << " [Drawing] Square at memory address: " << this << "\n";
}
};
//-------------------------------------------------- Concrete Shape Circle:
class Circle : public Shape
{
public:
Circle()
// Constructor
{
std::cout << " [Constructor] Created a new Circle\n";
}
Circle(const Circle& other) : Shape(other)
// Copy constructor
{
std::cout << " [Copy Constructor] Copied a Circle\n";
}
Circle(Circle&& other) noexcept : Shape(std::move(other))
{
std::cout << " [Move] Moved a Circle\n";
// Move constructor
}
void draw() const override
{
std::cout << " [Drawing] Circle at memory address: " << this << "\n";
}
};
// Define the variant to hold shapes by value
using ShapeVariant = std::variant<Square, Circle>;
//------------------------------------------------------------------- Main:
int main()
{
std::cout << "=== VARIANT & VALUE SEMANTICS: AUTOMATIC CLONING ===\n";
std::vector<ShapeVariant> originals;
std::vector<ShapeVariant> clones;
std::cout << "\n--- PHASE 1: Creating new objects ---\n";
originals.push_back(Square());
originals.push_back(Circle());
std::cout << "\n--- PHASE 2: Printing addresses in originals ---\n";
for(const auto& v : originals) {
std::visit([](const auto& shape) { shape.draw(); }, v);
}
std::cout << "\n--- PHASE 3: Copying into clones ---\n";
for(const auto& v : originals)
clones.push_back(v);
std::cout << "\n--- PHASE 4: Printing addresses in clones ---\n";
for (const auto& v : clones) {
std::visit([](const auto& shape) { shape.draw(); }, v);
}
std::cout << " [Success] All objects cloned automatically via value semantics.\n"
"           Compare the addresses above.\n";
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: StaticPolymorphism.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates Static Polymorphism using the CRTP idiom.
* It simulates a printer system where the 'Printer' base class defines
* the public interface, but the actual implementation is resolved at 
* compile-time by downcasting to the 'Derived' template argument.
* 
* --- KEY ADVANTAGE:
* We can write generic functions like 'executePrinter' that work with any 
* future printer type without using Virtual Tables (vtables). The compiler 
* generates optimized code for each specific printer type used.
* ============================================================================
*/
#include <iostream>
#include <string>
//--------------------------------------------------------- Base Class (CRTP):
template <class Derived>
class Printer
{
public:
// The public interface
void print(const std::string& message)
{
std::cout << " [Base] Preparing to print...\n";
// Static downcast: converting 'this' from Printer<Derived>* to Derived*
// This is safe because of the CRTP inheritance structure.
static_cast<Derived*>(this)->printImplementation(message);
}
};
//------------------------------------------------------- Concrete Printer A:
class InkjetPrinter : public Printer<InkjetPrinter>
{
public:
void printImplementation(const std::string& message)
{
std::cout << "  -> [Inkjet] Spraying ink: " << message << "\n";
}
};
//------------------------------------------------------- Concrete Printer B:
class LaserPrinter : public Printer<LaserPrinter>
{
public:
void printImplementation(const std::string& message)
{
std::cout << "  -> [Laser] Fusing toner: " << message << "\n";
}
};
//------------------------------------------------- Client Code (Static Dispatch):
// This function is the "contract". It can be written before knowing 
// all concrete printer implementations.
template <class T>
void executePrinter(Printer<T>* printer, const std::string& msg)
{
// Even through a base pointer, the correct derived method is called
// at compile-time without virtual table overhead.
printer->print(msg);
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== CRTP: STATIC POLYMORPHISM SIMULATION ===\n" << std::endl;
// 1. Using Inkjet Printer
InkjetPrinter* inkjet = new InkjetPrinter();
std::cout << "Testing Inkjet Printer:\n";
executePrinter(inkjet, "Medium Performance Printing");
std::cout << "\n";
// 2. Using Laser Printer
LaserPrinter* laser = new LaserPrinter();
std::cout << "Testing Laser Printer:\n";
executePrinter(laser, "High Performance Printing.");
// Cleanup
delete inkjet;
delete laser;
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: ObjectCounter.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the "Object Counter" idiom using CRTP. It allows 
* tracking the number of instances created and currently alive for any class.
* 
* --- THE POWER OF CRTP:
* Because 'ObjectCounter<X>' and 'ObjectCounter<Y>' are distinct types, 
* the compiler creates separate static variables for each. This ensures 
* that the count for 'Dragon' is independent of the count for 'Elephant',
* even though they share the same base logic.
* ============================================================================
*/
#include <iostream>
#include <string>
//--------------------------------------------------------- Counter Base (CRTP):
template <class T>
class ObjectCounter
{
private:
// C++17 inline static variables allow definition inside the header/class.
static inline int createdCount_{0};
static inline int aliveCount_{0};
protected:
// Protected constructor: only derived classes can instantiate.
ObjectCounter()
{
++createdCount_;
++aliveCount_;
}
// Destructor decrements only the alive count.
~ObjectCounter()
{
--aliveCount_;
}
public:
static int getCreated() { return createdCount_; }
static int getAlive()   { return aliveCount_; }
};
//------------------------------------------------------- Concrete Class A:
class Dragon : public ObjectCounter<Dragon>
{
public:
void roar() const { std::cout << " [Dragon] ROAR!\n"; }
};
//------------------------------------------------------- Concrete Class B:
class Elephant : public ObjectCounter<Elephant>
{
public:
void trumpet() const { std::cout << " [Elephant] TRUMPET!\n"; }
};
//----------------------------------------------------------- PrintSummary:
void printSummary()
{
std::cout << " Dragons created: " << Dragon::getCreated()
<< ", alive: " << Dragon::getAlive() << "\n";
std::cout << " Elephants created: " << Elephant::getCreated()
<< ", alive: " << Elephant::getAlive() << "\n\n";
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== CRTP: OBJECT COUNTER SIMULATION ===\n" << std::endl;
std::cout << "--- Initial state ---\n";
printSummary();
// 1. Creating objects
std::cout << "Creating 2 Dragons and 1 Elephant...\n";
Dragon* d1 = new Dragon();
Dragon* d2 = new Dragon();
Elephant* e1 = new Elephant();
printSummary();
// 2. Testing destruction
std::cout << "Destroying 1 Dragon...\n";
delete d1;
printSummary();
// Cleanup remaining
delete d2;
delete e1;
std::cout << "=== FINAL TOTALS ===\n";
printSummary();
std::cout << "=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: PolymorphicClone.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates a highly sophisticated, non-intrusive implementation 
* of the Prototype pattern using a combination of CRTP (Curiously Recurring 
* Template Pattern), Multiple Inheritance, and Cross-Casting.
* It demonstrates how CRTP can automate the Prototype pattern's 'clone()' 
* method and also the power of Multiple Inheritance to inject the "Clonable" 
* into a hierarchy with Mixin.
* 
* --- THE PROBLEM:
* In traditional polymorphism, every derived class must manually implement 
* its own 'clone()' method to return a copy of itself (see 04_Prototype). This
* leads to redundant, error-prone boilerplate code and tightly couples the
* business interface (e.g., 'Shape') to memory management concerns.
*
* --- THE SOLUTION (CRTP MIXIN):
* We use a generic 'Cloneable<T>' Mixin that implements the 'clone()' logic 
* once. By inheriting from this Mixin, concrete classes automatically gain 
* polymorphic cloning capabilities without writing any additional code. 
* This effectively automates the Prototype pattern.
*
* --- THE ARCHITECTURAL ADVANTAGE (NON-INTRUSIVE DESIGN):
* We separate Business Logic from System Infrastructure. The 'Shape' 
* interface remains 100% pure and unaware of cloning. This is achieved 
* through C++ Multiple Inheritance, allowing a class like 'Square' to 
* satisfy two independent hierarchies:
*    1. Business Hierarchy: Square "is a" Shape (for drawing).
*    2. System Hierarchy:   Square "is a" Cloneable object (for memory).
* 
* --- TECHNICAL MECHANICS (CROSS-CASTING):
* The system uses 'dynamic_cast' to perform a "cross-cast" between these 
* unrelated branches at runtime. When the client asks to clone a 'Shape', 
* the infrastructure ('Cloneable.h') searches the object's hierarchy for the 
* 'ICloneable' interface. If found, the CRTP-generated logic is executed.
* 
* This highlights a key advantage of C++ over languages like Java, as it 
* allows the clean injection of generic behaviors into independent class 
* hierarchies with zero overhead and absolute type safety.
* 
* --- ERROR HANDLING & SIMULATION:
* The infrastructure follows a "Fail-Fast" policy. 
* 1. Success Case: Cloning 'Square' and 'Circle' which implement the Mixin.
* 2. Error Case: Attempting to clone a 'Triangle' which is a valid 'Shape' 
*    but not 'Cloneable'. The system detects this at runtime and throws 
*    an exception, preventing null pointer propagation and ensuring 
*    architectural integrity.
* ============================================================================
*/
#include "Cloneable.h"
#include <iostream>
#include <vector>
#include <string>
//---------------------------------------------------- Abstract Base Shape:
class Shape
// Base
{
public:
virtual ~Shape() = default;
virtual void draw() const = 0;
};
//-------------------------------------------------- Concrete Shape Square:
class Square : public Shape, public Cloneable<Square>
{
public:
Square()
// Constructor
{
std::cout << " [Constructor] Created a new Square\n";
}
Square(const Square&)
// Copy constructor
{
std::cout << " [Copy Constructor] Copied a Square\n";
}
void draw() const override
{
std::cout << " [Drawing] Square at memory address: " << this << "\n";
}
};
//-------------------------------------------------- Concrete Shape Circle:
class Circle : public Shape, public Cloneable<Circle>
{
public:
Circle()
// Constructor
{
std::cout << " [Constructor] Created a new Circle\n";
}
Circle(const Circle&)
// Copy constructor
{
std::cout << " [Copy Constructor] Copied a Circle\n";
}
void draw() const override
{
std::cout << " [Drawing] Circle at memory address: " << this << "\n";
}
};
//------------------------------------------------ Concrete Shape Triangle:
// This class is NOT cloneable (missing the Cloneable Mixin by mistake)
class Triangle : public Shape
//, public Cloneable<Triangle>
{
public:
Triangle()
// Constructor
{
std::cout << " [Constructor] Created a new Triangle without the Cloneable Mixin\n";
}
Triangle(const Triangle&)
// Copy constructor
{
std::cout << " [Copy Constructor] Copying a Triangle without the Cloneable Mixin\n";
}
void draw() const override
{
std::cout << " [Drawing] Triangle at memory address: " << this << "\n";
}
};
//------------------------------------------------------------------- Main:
int main()
{
std::cout << "=== CRTP & CROSS-CASTING: SMART CLONING SYSTEM ===\n";
std::vector<std::unique_ptr<Shape>> originals;
std::vector<std::unique_ptr<Shape>> clones;
std::cout << "\n--- PHASE 1: Creating new objects ---\n";
originals.push_back(std::make_unique<Square>());
originals.push_back(std::make_unique<Circle>());
std::cout << "\n--- PHASE 2: Printing addresses in originals ---\n";
for(const auto& shape : originals) shape->draw();
try
{
std::cout << "\n--- PHASE 3: Copying into clones ---\n";
for(const auto& shape : originals)
clones.push_back(Cloneable_clone(shape.get()));
std::cout << "\n--- PHASE 4: Printing addresses in clones ---\n";
for (const auto& shape : clones) shape->draw();
std::cout << " [Success] All compatible objects cloned. Compare the addresses above.\n";
}
catch(const std::exception& e)
{
std::cerr << " [Unexpected Error] " << e.what() << "\n";
}
std::cout << "\n--- PHASE 5: Not cloneable error detection ---\n";
Triangle triangle;
triangle.draw();
try
{
std::cout << " Attempting to clone the Triangle...\n";
auto failed_clone = Cloneable_clone(&triangle);
}
catch (const std::exception& e)
{
std::cout << " [Caught Expected Error] " << e.what() << "\n";
}
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: VariantHeapClone.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates a hybrid approach: using std::variant for 
* heterogeneous storage while maintaining objects on the heap with 
* std::unique_ptr. This requires a custom cloning mechanism via CRTP.
* 
* --- THE PROBLEM:
* std::variant can hold std::unique_ptr, but since unique_ptr is non-copyable, 
* the entire variant becomes non-copyable. Standard value semantics cloning 
* (clones = originals) will no longer compile.
* 
* --- THE SOLUTION (CRTP MIXIN + VISIT):
* 1. We re-introduce the Cloneable<T> Mixin to provide a non-virtual clone() 
*    method that returns a new std::unique_ptr<T>.
* 2. We use std::visit during the cloning phase to dispatch the call to 
*    the correct CRTP clone() implementation at compile-time.
* 
* --- ADVANTAGES:
* 1. Performance: Zero virtual function overhead for cloning.
* 2. Safety: No raw pointers or manual memory management.
* 3. Architecture: Business logic (Shape) remains separate from memory 
*    concerns (Cloneable).
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <variant>
//--------------------------------------------------------- Cloneable Mixin:
template <typename Concrete>
class Cloneable {
public:
std::unique_ptr<Concrete> clone() const {
return std::make_unique<Concrete>(static_cast<const Concrete&>(*this));
}
};
//---------------------------------------------------- Abstract Base Shape:
class Shape
// Base
{
public:
virtual ~Shape() = default;
virtual void draw() const = 0;
};
//-------------------------------------------------- Concrete Shape Square:
class Square : public Shape, public Cloneable<Square>
{
public:
Square()
// Constructor
{
std::cout << " [Constructor] Created a new Square on the Heap\n";
}
Square(const Square& other) : Shape(other)
// Copy constructor
{
std::cout << " [Copy Constructor] Copied a Square\n";
}
void draw() const override
{
std::cout << " [Drawing] Square at memory address: " << this << "\n";
}
};
//-------------------------------------------------- Concrete Shape Circle:
class Circle : public Shape, public Cloneable<Circle>
{
public:
Circle()
// Constructor
{
std::cout << " [Constructor] Created a new Circle on the Heap\n";
}
Circle(const Circle& other) : Shape(other)
// Copy constructor
{
std::cout << " [Copy Constructor] Copied a Circle\n";
}
void draw() const override
{
std::cout << " [Drawing] Circle at memory address: " << this << "\n";
}
};
// Define the variant to hold shapes by unique_ptr
using ShapeVariant = std::variant<std::unique_ptr<Square>, std::unique_ptr<Circle>>;
//------------------------------------------------------------------- Main:
int main()
{
std::cout << "=== VARIANT & HEAP STORAGE: CRTP CLONING SYSTEM ===\n";
std::vector<ShapeVariant> originals;
std::vector<ShapeVariant> clones;
std::cout << "\n--- PHASE 1: Creating new objects ---\n";
originals.push_back(std::make_unique<Square>());
originals.push_back(std::make_unique<Circle>());
std::cout << "\n--- PHASE 2: Printing addresses in originals ---\n";
for(const auto& v : originals) {
std::visit([](const auto& shapePtr) { shapePtr->draw(); }, v);
}
std::cout << "\n--- PHASE 3: Copying into clones ---\n";
// We must manually clone because unique_ptr cannot be copied.
// std::visit finds the correct clone() method at compile-time.
for(const auto& v : originals) {
auto clonedVariant = std::visit([](const auto& shapePtr) -> ShapeVariant {
return shapePtr->clone();
// Note: The '-> ShapeVariant' forces all lambda paths to return the same type.
// Alternatively, one could write:
//return ShapeVariant(shapePtr->clone());
}, v);
clones.push_back(std::move(clonedVariant));
}
std::cout << "\n--- PHASE 4: Printing addresses in clones ---\n";
for (const auto& v : clones) {
std::visit([](const auto& shapePtr) { shapePtr->draw(); }, v);
}
std::cout << " [Success] All heap-allocated objects cloned via CRTP Mixin.\n"
"           Compare the addresses above.\n";
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: ErrorHandling_Variant.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the "Either" pattern using std::variant (C++17).
* It represents a safe way to handle errors without using exceptions or 
* ambiguous return codes.
* 
* --- THE SCENARIO:
* A function attempts to parse a string into a number and then calculates 
* its square root. There are two failure points: 
* 1. The input is not a number (Parsing error).
* 2. The number is negative (Mathematical error).
* 
* --- THE VISITOR APPROACH:
* Using std::visit ensures that the caller MUST handle both the success 
* and the error case, providing total type safety at runtime.
* ============================================================================
*/
#include <iostream>
#include <variant>
#include <string>
#include <cmath>
#include <iomanip>
//--------------------------------------------------------- Error Structure:
struct Error
{
std::string message;
};
//--------------------------------------------------------- Result Type:
// A Result can be either the expected double or an Error object.
using Result = std::variant<double, Error>;
//--------------------------------------------------------- Logic:
Result safeSqrt(const std::string& input)
{
double value;
// 1. Try to parse the string
try 
{
size_t pos;
value = std::stod(input, &pos);
// Ensure the whole string was a number
if (pos < input.size()) return Error{"Input contains non-numeric characters."};
}
catch (...)
{
return Error{"Invalid input: '" + input + "' is not a number."};
}
// 2. Check for mathematical validity
if (value < 0) return Error{"Math error: cannot calculate sqrt of " + std::to_string(value)};
// 3. Success
return std::sqrt(value);
}
//--------------------------------------------------------- Result Processor:
struct ResultHandler
{
void operator()(double value) const
{
std::cout << " [Success] Result: " << std::fixed << std::setprecision(4) << value << "\n";
}
void operator()(const Error& error) const
{
std::cout << " [Failure] " << error.message << "\n";
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== ERROR HANDLING (std::variant VERSION) ===\n" << std::endl;
std::string testInputs[] = {"16.0", "-4.0", "abc", "25.0"};
for (const auto& input : testInputs)
{
std::cout << "Calculating sqrt of '" << input << "':\n";
// The function returns a Variant (The "Either" pattern)
Result res = safeSqrt(input);
// Process the variant using a Visitor
std::visit(ResultHandler{}, res);
std::cout << std::endl;
}
std::cout << "=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: ErrorHandling_Expected.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates the "Result" pattern using std::expected (C++23).
* This is the industry-standard way to handle operations that can fail 
* without resorting to exceptions for logic flow.
* 
* --- ADVANTAGES OVER std::variant:
* 1. Semantic Clarity: The type signature 'expected<double, Error>' 
*    explicitly states what is the result and what is the failure.
* 2. Ergonomics: Provides methods like .has_value(), .value(), and .error() 
*    which make the code much more readable than a Visitor.
* 3. Functional Style: It behaves like a pointer (bool conversion and 
*    operator*) but with total type safety.
* ============================================================================
*/
#include <iostream>
#include <expected>
// C++23 Feature
#include <string>
#include <cmath>
#include <iomanip>
//--------------------------------------------------------- Error Structure:
struct Error
{
std::string message;
};
//--------------------------------------------------------- Logic:
// The function signature is now extremely expressive.
std::expected<double, Error> safeSqrt(const std::string& input)
{
double value;
// 1. Try to parse the string
try 
{
size_t pos;
value = std::stod(input, &pos);
if (pos < input.size()) return std::unexpected(Error{"Input contains non-numeric characters."});
}
catch (...)
{
return std::unexpected(Error{"Invalid input: '" + input + "' is not a number."});
}
// 2. Check for mathematical validity
if (value < 0) return std::unexpected(Error{"Math error: cannot calculate sqrt of " + std::to_string(value)});
// 3. Success: Just return the value
return std::sqrt(value);
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== ERROR HANDLING (std::expected VERSION - C++23) ===\n" << std::endl;
std::string testInputs[] = {"16.0", "-4.0", "abc", "25.0"};
for (const auto& input : testInputs)
{
std::cout << "Calculating sqrt of '" << input << "':\n";
// The function returns an 'expected' object
auto res = safeSqrt(input);
// Checking success is as simple as checking a boolean
if (res)
std::cout << " [Success] Result: " << std::fixed << std::setprecision(4) << *res << "\n";
else
std::cout << " [Failure] " << res.error().message << "\n";
std::cout << std::endl;
}
std::cout << "=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: ThreadPool_Example.cpp
* Author: Mario Galindo Queralt, Ph.D.
* 
* --- DESIGN OVERVIEW:
* This program demonstrates an advanced asynchronous pipeline using a
* task-agnostic ThreadPool. It simulates a two-stage scientific workflow where
* workers process complex data structures and return results along with the
* original inputs for full traceability.
* 
* --- BATCH PROCESSING LADDER:
* 1. Batch 1: Single-input/Single-output (Square Roots).
* 2. Batch 2: Multi-input/Multi-output (Sines and Cosines from dual inputs).
* 
* --- ARCHITECTURAL SEPARATION:
* - Computation Logic: Pure functions (my_sqrt, my_trig) that encapsulate the
*   heavy mathematical work, independent of the threading model.
* - Dispatch Logic: Flexible lambdas that act as "glue", bridging the
*   ThreadPool with SafeQueues to coordinate inputs and results.
* - Reporting Logic: Ephemeral reporter threads created per batch to handle
*   specific I/O requirements and persistence to disk.
* 
* --- CONCURRENCY TOPOLOGY & SYNCHRONIZATION:
* - Producer Thread: Generates problems and submits them to the pool.
* - Worker Threads (15): Persistent threads that consume and execute tasks.
* - Reporter Thread: Consumes results and persists them to disk.
* - Determinism: We use explicit producer joining to ensure all tasks are
*   submitted and the 'wait_until_all_tasks_are_processed()' barrier to ensure
*   all tasks are processed before the reporter shuts down, guaranteeing data
*   integrity across transitions.
*   ============================================================================
*/
#include "ThreadPool.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>
//--------------------------------------------------------- Functional Structs:
// Results returned by the heavy computation functions
struct SqrtOutput
{
double val;
};
struct TrigOutput
{
double sinVal;
double cosVal;
};
//--------------------------------------------------------- Queue Structs:
// Packets that travel through the result queues
struct ResultRoots
{
int    id;
double input;
double value;
};
struct ResultTrig
{
int    id;
double input1;
double input2;
double sinVal;
double cosVal;
};
//--------------------------------------------------------- Heavy Computations:
void simulate_work(int min, int max)
{
static std::random_device rd;
static std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(min, max);
std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
}
SqrtOutput my_sqrt(double n)
{
simulate_work(50, 350);
// Simulate complex root finding
return { std::sqrt(n) };
}
TrigOutput my_trig(double n1, double n2)
{
simulate_work(90, 450);
// Simulate complex trigonometric analysis
return { std::sin(n1), std::cos(n2) };
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== THREAD POOL: ASYNCHRONOUS PIPELINE SIMULATION ===\n" << std::endl;
// Create a pool of working threads
ThreadPool pool(15, 20);
// num_workers = 15, task_max_queue_size = 20
// ---------------------------------------------------------------- PHASE 1: Square Roots:
{
std::cout << "--- Starting Batch 1: Square Roots ---\n";
// Create a queue to receive the results.
SafeQueue<ResultRoots> root_results_queue{10};
// max_queue_size = 10
// Create a thread to get the results and use them as appropriate.
std::jthread reporter([&root_results_queue]()
{
std::ofstream file("batch1_roots.txt");
ResultRoots result_root;
// Get results
while(root_results_queue.pop(result_root))
{
// Use them as appropriate
file << "Job ID "
<< result_root.id
<< ": sqrt("
<< result_root.input
<< ") = "
<< result_root.value
<< "\n";
}
std::cout << " [Reporter 1] All roots saved. Closing file.\n";
});
// Create a thread to produce the data and send it for processing
std::jthread producer([&pool, &root_results_queue]()
{
for(int i = 1; i <= 200; ++i)
{
// Prepare input data
simulate_work(0, 10);
double val = static_cast<double>(i);
// Send a function with the data to be processed
pool.submit([&root_results_queue, i, val]()
{
// This is the work that the worker will perform
SqrtOutput res = my_sqrt(val);
// This is how the worker will return the results
root_results_queue.push({
.id=i,
.input=val,
.value=res.val
});
});
}
std::cout << " [Producer 1] All jobs submitted.\n";
});
// Wait for producer thread to end producing all tasks
producer.join();
// Wait until all tasks have been processed by the worker threads
pool.wait_until_all_tasks_are_processed();
// Close the results queue, thus stopping the reporter
root_results_queue.close();
simulate_work(5, 5);
std::cout << " [System] Batch 1 completed.\n";
}
// --------------------------------------------------------------- PHASE 2: Sine & Cosine:
{
std::cout << "\n--- Starting Batch 2: Sine & Cosine ---\n";
// Create a queue to receive the results.
SafeQueue<ResultTrig> trig_results_queue{10};
// queue_size = 10
// Create a thread to get the results and use them as appropriate.
std::jthread reporter([&trig_results_queue]()
{
std::ofstream fileSin("batch2_sines.txt");
std::ofstream fileCos("batch2_cosines.txt");
ResultTrig result_trig;
// Get results
while(trig_results_queue.pop(result_trig))
{
// Use them as appropriate
fileSin << "Job ID "
<< result_trig.id
<< ": sin("
<< result_trig.input1
<< ") = "
<< result_trig.sinVal
<< "\n";
fileCos << "Job ID "
<< result_trig.id
<< ": cos("
<< result_trig.input2
<< ") = "
<< result_trig.cosVal
<< "\n";
}
std::cout << " [Reporter 2] Trig results saved. Closing files.\n";
});
// Create a thread to produce the data and send it for processing
std::jthread producer([&pool, &trig_results_queue]()
{
for(int i = 1; i <= 100; ++i)
{
// Prepare input data
simulate_work(0, 15);
double val1 = static_cast<double>(i) * 0.1;
double val2 = static_cast<double>(i) * 0.2;
// Send a function with the data to be processed
pool.submit([&trig_results_queue, i, val1, val2]()
{
// This is the work that the worker will perform
TrigOutput res = my_trig(val1, val2);
// This is how the worker will return the results
trig_results_queue.push({
.id = i,
.input1 = val1,
.input2 = val2,
.sinVal = res.sinVal,
.cosVal = res.cosVal
});
});
}
std::cout << " [Producer 2] All jobs submitted.\n";
});
// Wait for producer thread to end producing all tasks
producer.join();
// Wait until all tasks have been processed by the worker threads
pool.wait_until_all_tasks_are_processed();
// Close the results queue, thus stopping the reporter
trig_results_queue.close();
simulate_work(5, 5);
std::cout << " [System] Batch 2 completed.\n";
}
std::cout << "\n=== SIMULATION COMPLETED. SHUTTING DOWN POOL ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Polymorphic_bottleneck.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program serves as a performance baseline analysis to evaluate the real
* computational cost of Dynamic Dispatch (Virtual Functions) under different
* memory organization layouts. It contrasts classical Object-Oriented (OOP)
* polymorphism against Data-Oriented Design (DOD) principles.
*
* --- DYNAMIC DISPATCH & HARDWARE COOPERATION:
* Classical C++ polymorphism relies on runtime vtable lookups. However, modern
* CPU architectures mitigate this overhead via the Branch Target Buffer (BTB).
* This program contrasts virtual dispatch against Data-Oriented Design (DOD)
* principles to prove a fundamental rule:
* 1. Predictable Execution: When a virtual function is invoked sequentially
*    with a predictable type pattern, the hardware branch predictor caches the
*    destination perfectly. This avoids branch misprediction stalls, isolating
*    the unavoidable baseline overhead of vtable indirection (pointer chasing).
* 2. Chaotic Execution: When pointers to heterogeneous types are interleaved
*    randomly, the hardware predictor falls into a permanent state of
*    misprediction, forcing massive CPU pipeline flushes. Additionally,
*    shuffled pointer access triggers frequent 'Cache Misses'.
* 3. Concrete (DOD) Execution: When invoking non-virtual (concrete) methods on
*    homogeneously grouped objects, we bypass both indirection and branch
*    prediction entirely. This enables aggressive compiler optimizations (like
*    inlining) and maximizes instruction cache locality, delivering the highest
*    possible hardware execution throughput.
*
* --- COMPILER TIMING SAFEGUARDS:
* High-level optimization flags (-O3) tend to pre-calculate or eliminate
* loops that accumulate constant values at compile-time. To ensure honest
* hardware profiling without assembly-level trickery, this implementation:
* 1. Employs 'volatile' type qualifiers on accumulators to enforce strict
*    read/write operations on memory per iteration.
* 2. All functions return the same constant (7) to ensure the ALU performs
*    the exact same mathematical workload in every test case.
* ============================================================================
*/
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
//-------------------------------------------- CPU Warmup via FPU workload:
// Performs floating-point square root operations to force the CPU 
// out of low-power idle states (C-states) into its peak turbo frequency.
void warm_up_cpu(size_t iterations)
{
std::cout << "--- Warming up the CPU ---\n" << std::endl;
volatile double accumulator = 0.0;
for (size_t i = 0; i < iterations; ++i)
{
accumulator += std::sqrt(static_cast<double>(i));
}
}
//--------------------------------------------------------- Test Hierarchy:
class Base
{
public:
int concrete() const { return 7; }
virtual int virt() const = 0;
virtual ~Base() = default;
};
class DerivedA : public Base
{
public:
int virt() const override { return 7; }
};
class DerivedB : public Base
{
public:
int virt() const override { return 7; }
};
//--------------------------------------------------------- Main Simulation:
int main()
{
// WORKLOAD BOUNDS: 1,000,000,000 operations total per test.
// num_iterations = num_objects * num_passes
const size_t num_objects    = 10'000;
const size_t num_passes     = 100'000;
const size_t num_warm_cpu   = 1'000'000'000;
std::cout << "=== POLYMORPHIC BOTTLENECK ANALYSIS ===\n" << std::endl;
//--------------------------------------------------------------------------
// TEST 1: PREDICTABLE MEMORY ACCESS (Orderly Vector Layout)
// Hardware can predict the virtual call destination perfectly.
//--------------------------------------------------------------------------
std::cout << "--- [TEST 1] PREDICTABLE ACCESS (Orderly Object) ---" << std::endl;
{
std::vector<Base*> predictable_objects;
predictable_objects.reserve(num_objects);
for(size_t i = 0; i < num_objects; ++i)
predictable_objects.push_back(new DerivedA());
warm_up_cpu(num_warm_cpu);
// Measure Concrete Call (Sequential pointer access, but same method)
volatile size_t total_concrete = 0;
auto start = std::chrono::steady_clock::now();
for(size_t pass = 0; pass < num_passes; ++pass)
for(size_t i = 0; i < num_objects; ++i)
total_concrete += predictable_objects[i]->concrete();
auto end = std::chrono::steady_clock::now();
auto t_concrete = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
// Measure Virtual Call (Sequential pointer access with predictable vtable jumps)
volatile size_t total_virt = 0;
start = std::chrono::steady_clock::now();
for(size_t pass = 0; pass < num_passes; ++pass)
for(size_t i = 0; i < num_objects; ++i)
total_virt += predictable_objects[i]->virt();
end = std::chrono::steady_clock::now();
auto t_virtual = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
std::cout << " Concrete function: " << t_concrete << " ms" << std::endl;
std::cout << " Virtual function : " << t_virtual << " ms" << std::endl;
std::cout << " Delta Overhead   : " << (t_virtual - t_concrete)
<< " ms (Predictable Virtual Overhead!)\n" << std::endl;
for(auto obj : predictable_objects) delete obj;
}
//--------------------------------------------------------------------------
// TEST 2: CHAOTIC MEMORY ACCESS (Shuffled Vector Layout)
// Forces Branch Mispredictions and Cache Misses.
//--------------------------------------------------------------------------
std::cout << "--- [TEST 2] CHAOTIC ACCESS (Shuffled Layout) ---" << std::endl;
{
std::vector<Base*> chaotic_objects;
chaotic_objects.reserve(num_objects);
for(size_t i = 0; i < num_objects / 2; ++i)
{
chaotic_objects.push_back(new DerivedA());
chaotic_objects.push_back(new DerivedB());
}
// Randomize the order to destroy hardware predictability
std::random_device rd;
std::mt19937 g(rd());
std::shuffle(chaotic_objects.begin(), chaotic_objects.end(), g);
warm_up_cpu(num_warm_cpu);
// Measure Concrete Call (Sequential pointer access, but same method)
volatile size_t total_concrete = 0;
auto start = std::chrono::steady_clock::now();
for(size_t pass = 0; pass < num_passes; ++pass)
for(size_t i = 0; i < num_objects; ++i)
total_concrete += chaotic_objects[i]->concrete();
auto end = std::chrono::steady_clock::now();
auto t_concrete = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
// Measure Virtual Call (Sequential pointer access, but chaotic vtable jumps)
volatile size_t total_virt = 0;
start = std::chrono::steady_clock::now();
for(size_t pass = 0; pass < num_passes; ++pass)
for(size_t i = 0; i < num_objects; ++i)
total_virt += chaotic_objects[i]->virt();
end = std::chrono::steady_clock::now();
auto t_virtual = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
std::cout << " Concrete function: " << t_concrete << " ms" << std::endl;
std::cout << " Virtual function : " << t_virtual << " ms" << std::endl;
std::cout << " Delta Overhead   : " << (t_virtual - t_concrete)
<< " ms (MASSIVE PENALTY!)\n" << std::endl;
for(auto obj : chaotic_objects) delete obj;
}
std::cout << "=== ANALYSIS COMPLETED ===" << std::endl;
}
//============================================================================== END
/**
* ============================================================================
* File: Component.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program demonstrates an advanced Entity-Component-System (ECS)
* architecture built on Data-Oriented Design (DOD) principles. It represents
* a shift from traditional Object-Oriented programming to a model that
* prioritizes memory layout and CPU cache efficiency.
*
* --- STATIC TYPE-TO-STORAGE MAPPING:
* Instead of using inheritance-based lists or dynamic maps of pointers, this
* architecture utilizes C++ templates to map each unique 'ComponentType'
* to its own 'static std::vector'. This provides:
* 1. Zero-Overhead: Component retrieval is resolved at compile-time,
*    eliminating runtime lookups and virtual function (vtable) overhead.
* 2. Type Safety: The system is intrinsically type-safe; the compiler
*    guarantees that you always receive the exact vector type requested.
*
* --- ENTITY IDENTITY & UNIQUENESS:
* Entities are not objects, but simple 'uint32_t' identifiers. The Registry
* (acting as the World Container) manages an internal counter through
* 'createEntity()' to guarantee that every entity receives a strictly
* unique ID, preventing identity collisions in the global data buckets.
*
* --- PERFORMANCE & CACHE LOCALITY:
* Components are stored in contiguous memory blocks. This is vital because:
* 1. Cache Locality: Systems (like Physics or AI) process these arrays
*    sequentially. The CPU can effectively prefetch data, drastically
*    reducing 'Cache Misses'.
* 2. Data Normalization: Mathematical components (Position, Velocity) are
*    kept "lean" (small PODs) to fit more elements into a single CPU cache
*    line. Human-readable names or metadata are relegated to a separate
*    'Label' component, processed only when necessary (e.g., during logging).
*
* --- PARALLEL ARRAY BUILDER:
* To ensure data integrity, we utilize a Static Builder pattern (Pattern 01).
* This builder guarantees "Parallel Array Alignment": every time an entity
* is built, a new entry is pushed into EVERY component vector simultaneously.
* This allows Systems to use a shared index 'i' to access related data across
* different buckets in O(1) time without searching for IDs.
*
* --- SINGLETON ARCHITECTURE:
* The 'Registry' is implemented as a Meyers Singleton. This centralizes
* identity management and data storage, ensuring a single, consistent
* "World State" across the entire application. It prevents data
* fragmentation and provides a global access point for specialized Systems.
*
* --- DYNAMIC STATE SIMULATION:
* In DOD, objects don't "decide" to change state. Instead, logic systems or
* game controllers mutate the data in the vectors. Behavioural changes are
* the side effect of updated values being processed in the next cycle.
* ============================================================================
*/
#include <vector>
// Provides contiguous memory storage for the static component buckets
#include <iostream>
// Used for simulation reporting and console output
#include <type_traits>
// Required for std::decay_t to strip references and const from types
#include <string>
// Used for entity labels and human-readable metadata
#include <cstdint>
// Provides fixed-width integer types like uint32_t for entity IDs
#include <utility>
// Required for std::forward (perfect forwarding) and std::move
//--------------------------------------------------------- Component Types:
// Pure data structures (POD) for maximum cache efficiency.
struct Label     { uint32_t entityId; std::string name; };
struct Position  { uint32_t entityId; float x, y; };
struct Velocity  { uint32_t entityId; float vx, vy; };
struct AIControl { uint32_t entityId; int state; };
// 0:Idle, 1:Patrol, 2:Attack, 3:Defense, 4:Dead
//--------------------------------------------------------- Component Registry:
class Registry
// A Meyers' Singleton class
{
private:
uint32_t nextEntityId_{1};
Registry()                = default;
// 1:DC - Default Constructor
Registry(const Registry&) = delete;
// 2:CC - Copy Constructor
// Internal storage: one static vector per unique component type.
// We use this second template to ensure that even if getComponents is called
// with references or const types, they all map to the same physical vector.
template<class ComponentType>
std::vector<ComponentType>& getInternalVector()
{
static std::vector<ComponentType> componentVector;
return componentVector;
}
public:
// Registry is a Singleton
static Registry& getInstance()
{
static Registry instance;
return instance;
}
// Implementation of the Inernal Static Builder for Entity Creation
class EntityBuilder
{
private:
uint32_t    id_;
std::string name_{"Unknown"};
float       x_{0}, y_{0}, vx_{0}, vy_{0};
int         state_{0};
// Private constructor: ensures only Registry can start the building process.
// This protects the integrity of the nextEntityId_ counter.
explicit EntityBuilder(uint32_t id) : id_{id} { }
// We grant friendship to the outer class
friend class Registry;
public:
EntityBuilder& setName(std::string name)       { name_ = std::move(name); return *this; }
EntityBuilder& setPosition(float x, float y)   { x_ = x; y_ = y;          return *this; }
EntityBuilder& setVelocity(float vx, float vy) { vx_ = vx; vy_ = vy;      return *this; }
EntityBuilder& setAIState(int state)           { state_ = state;          return *this; }
// The build method ensures all parallel arrays are updated at once.
// This alignment is what allows O(1) access by index in the systems.
//
// DESIGN NOTE: To preserve perfect array alignment, every entity gets a slot
// in every vector. If an entity does not require a specific component (e.g., 
// a static obstacle needing no AI or Velocity), it is possible to push a default
// "Null Object" or "Sentinel" component representing an inactive state, not
// implemented in this example. This technique avoids the structural complexity
// of sparse-set arrays while keeping memory access O(1).
uint32_t build()
{
auto& world = Registry::getInstance();
world.addComponent(Label{id_, std::move(name_)});
world.addComponent(Position{id_, x_, y_});
world.addComponent(Velocity{id_, vx_, vy_});
world.addComponent(AIControl{id_, state_});
return id_;
}
};
// Entry point to start the fluent building process
EntityBuilder createEntity()
{
return EntityBuilder{nextEntityId_++};
}
// This is the only way to get Components
template<class ComponentType>
std::vector<std::decay_t<ComponentType>>& getComponents()
{
return getInternalVector<std::decay_t<ComponentType>>();
}
private:
// Only Builder can add a Component
template<class ComponentType>
void addComponent(ComponentType&& component)
{
getComponents<ComponentType>().push_back(std::forward<ComponentType>(component));
}
};
//--------------------------------------------------------- Systems:
// ScenarioSystem: Handles the world timeline by identifying entities by their Labels.
class ScenarioSystem
{
public:
void update(int frame) const
{
auto& world    = Registry::getInstance();
auto& aiStates = world.getComponents<AIControl>();
auto& labels   = world.getComponents<Label>();
// Data mutation logic based on aligned indices
// In a real simulation, this system would analyze the environment
// (proximity, line of sight, health) to trigger state changes.
// For this example, we simulate these triggers based on the frame timeline.
for(size_t i = 0; i < labels.size(); ++i)
{
// Logic for Mario
if(labels[i].name == "Mario")
{
if(frame == 3) aiStates[i].state = 3;
// Mario detects danger and shields
if(frame == 5) aiStates[i].state = 0;
// Threat neutralized, back to Idle
}
// Logic for the Drone
if(labels[i].name == "Aggressive Drone")
{
if(frame == 2) aiStates[i].state = 2;
// Drone starts attack
if(frame == 4) aiStates[i].state = 4;
// Drone receives critical damage
}
}
}
};
class PhysicsSystem
{
public:
void update() const
{
auto& world      = Registry::getInstance();
auto& positions  = world.getComponents<Position>();
auto& velocities = world.getComponents<Velocity>();
auto& aiStates   = world.getComponents<AIControl>();
auto& labels     = world.getComponents<Label>();
std::cout << " [System] Updating Physics...\n";
// Using shared index 'i' for O(1) access to parallel arrays
for(size_t i = 0; i < positions.size(); ++i)
{
if(aiStates[i].state == 4) continue;
// Skip processing for dead entities
positions[i].x += velocities[i].vx;
positions[i].y += velocities[i].vy;
std::cout << "  -> [" << labels[i].name << "] moved to ("
<< positions[i].x << ", " << positions[i].y << ")\n";
}
}
};
/**
* AISystem along with ScenarioSystem, manages the intelligence for
* decision-making regarding game events. In a complex application, these
* systems can be implemented using Artificial Intelligence (AI) algorithms
* (e.g., Finite State Machines, Behavior Trees, or Neural Networks) to
* analyze data buckets and determine tactical actions.
*/
class AISystem
{
public:
void update() const
{
auto& world    = Registry::getInstance();
auto& aiStates = world.getComponents<AIControl>();
auto& labels   = world.getComponents<Label>();
std::cout << " [System] Updating AI Decisions...\n";
for(size_t i = 0; i < aiStates.size(); ++i)
{
const std::string& name = labels[i].name;
switch(aiStates[i].state)
{
case 0: std::cout << "  -> [" << name << "] is searching for the treasure.\n";                break;
case 1: std::cout << "  -> [" << name << "] is patrolling the area.\n";                       break;
case 2: std::cout << "  -> [" << name << "] is ATTACKING!\n";                                 break;
case 3: std::cout << "  -> [" << name << "] is in DEFENSE stance (Shields Up and Firing).\n"; break;
case 4: std::cout << "  -> [" << name << "] HAS CRASHED (Dead).\n";                           break;
}
}
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== COMPONENT REGISTRY (DOD / SINGLETON / BUILDER) ===\n" << std::endl;
Registry& world = Registry::getInstance();
// 1. Create Entities using the Fluent Static Builder (Pattern 01)
// This ensures all parallel vectors are perfectly aligned.
[[maybe_unused]]
auto mario = world.createEntity()
.setName("Mario")
.setPosition(0.0f, 0.0f)
.setVelocity(1.0f, 2.0f)
.setAIState(0)
.build();
[[maybe_unused]]
auto guard = world.createEntity()
.setName("Enemy Guard")
.setPosition(10.0f, 8.0f)
.setVelocity(-1.5f, 1.0f)
.setAIState(1)
.build();
[[maybe_unused]]
auto drone = world.createEntity()
.setName("Aggressive Drone")
.setPosition(7.0f, 2.5f)
.setVelocity(-2.0f, 1.0f)
.setAIState(1)
.build();
// 2. Systems Initialization
ScenarioSystem scenario;
PhysicsSystem  physics;
AISystem       intelligence;
// 3. Main Loop: Processing 5 frames of aligned simulation
for(int frame = 1; frame <= 5; ++frame)
{
std::cout << "--- Processing Frame " << frame << " ---\n";
// The ScenarioSystem manages data mutations for this frame
scenario.update(frame);
// The PhysicsSystem manages the physical simulation
physics.update();
// The AISystem manages the game's artificial intelligence
intelligence.update();
std::cout << std::endl;
}
std::cout << "=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Serialization.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program demonstrates the Type Erasure pattern. It achieves "Open
* Polymorphism" by allowing unrelated classes to be treated as a single
* type without requiring a common base class (non-intrusive).
*
* --- C++20 CONCEPTS:
* We use a formal compile-time contract (concept) to ensure that only classes
* providing a 'serialize' method can be wrapped, improving compiler error
* messages without adding runtime overhead.
*
* --- THE ARCHITECTURAL TRIAD:
* 1. The Wrapper (SerializableEntity): The public-facing class that the
*    user interacts with. It manages the "Rule of Seven".
* 2. The StorageInterface (Base): A private, internal abstract interface that
*    defines the behavior to be erased.
* 3. The Model (Template): A private, internal template that implements
*    the StorageInterface for any specific type (BusinessType).
*
* --- THE RULE OF SEVEN:
* Type Erasure is the ultimate test for object lifecycle management. We
* implement all seven key methods to ensure deep copies, efficient moves,
* and safe destruction of the opaque internal pointer.
* ============================================================================
*/
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>
#include <concepts>
//--------------------------------------------------------- 0. Formal Contract:
// Defines what it means to be "Serializable" at compile-time.
template <class T>
concept Serializable = requires(const T& object, std::ostream& os)
{
object.serialize(os);
};
//--------------------------------------------------------- 1. Business classes:
// These classes are 100% independent. No inheritance, no virtual methods.
class User
{
private:
std::string name_;
public:
explicit User(std::string name) : name_{std::move(name)} { }
void serialize(std::ostream& os) const
{
os << "{ \"user\": \"" << name_ << "\" }";
}
};
class NetworkConfig
{
private:
int port_;
public:
explicit NetworkConfig(int port) : port_{port} { }
void serialize(std::ostream& os) const
{
os << "{ \"port\": " << port_ << " }";
}
};
//--------------------------------------------------------- 2. Type erasure container:
class SerializableEntity
{
private:
// --- Internal Infrastructure ---
class StorageInterface
{
public:
virtual ~StorageInterface() = default;
virtual void serialize(std::ostream& os)          const = 0;
virtual std::unique_ptr<StorageInterface> clone() const = 0;
};
// Only accepts types that satisfy the 'Serializable' concept.
template <Serializable BusinessType>
class Model final : public StorageInterface
{
private:
BusinessType data_;
public:
explicit Model(BusinessType&& value) : data_{std::move(value)} { }
explicit Model(const BusinessType& value) : data_{value} { }
// --- Domain Operation (Internal/Contractual) ---
// Delegates directly to the business class. The wrapped type must natively 
// implement this method to satisfy the compile-time 'Serializable' concept.
void serialize(std::ostream& os) const override
{
data_.serialize(os);
}
// --- Extraneous Extension (External/Non-Intrusive) ---
// This is an external capability injected solely by the Type Erasure wrapper.
// The business classes are completely unaware of 'clone()', demonstrating how 
// this pattern allows adding new polymorphic behaviors without altering the original classes.
std::unique_ptr<StorageInterface> clone() const override
{
return std::unique_ptr< Model<BusinessType> >(new Model<BusinessType>(data_));
}
};
std::unique_ptr<StorageInterface> pimpl_;
public:
// --- APPLYING THE RULE OF SEVEN ---
// 1:DC - Default Constructor: Disabled.
SerializableEntity() = delete;
// 2:CC - Copy Constructor: Deep copy via virtual clone()
SerializableEntity(const SerializableEntity& other)
: pimpl_{other.pimpl_ ? other.pimpl_->clone() : nullptr}
{
std::cout << " [Rule of Seven] 2:CC - Copy Constructor (Deep Copy).\n";
}
// 3:MC - Move Constructor: Zero-cost pointer transfer
SerializableEntity(SerializableEntity&& other) noexcept : pimpl_{std::move(other.pimpl_)}
{
std::cout << " [Rule of Seven] 3:MC - Move Constructor.\n";
}
// 4:CA - Copy Assignment: Copy-and-Swap idiom for strong exception safety
SerializableEntity& operator=(const SerializableEntity& other)
{
std::cout << " [Rule of Seven] 4:CA - Copy Assignment.\n";
if(this != &other)
{
SerializableEntity temp(other);
// 2:CC
std::swap(pimpl_, temp.pimpl_);
}
return *this;
}
// 5:MA - Move Assignment: Efficient ownership transfer
SerializableEntity& operator=(SerializableEntity&& other) noexcept
{
std::cout << " [Rule of Seven] 5:MA - Move Assignment.\n";
if(this != &other) pimpl_ = std::move(other.pimpl_);
return *this;
}
// 6:De - Destructor
~SerializableEntity()
{
if(pimpl_) std::cout << " [Rule of Seven] 6:De - Destructor (Internal pimpl released).\n";
}
// 7:PC - Parametric Constructor (Template): Captures any Serializable type.
// Only accepts types that satisfy the 'Serializable' concept.
template <Serializable BusinessType>
SerializableEntity(BusinessType&& object)
: pimpl_{std::make_unique< Model<std::decay_t<BusinessType>> >(std::forward<BusinessType>(object))}
{
std::cout << " [Rule of Seven] 7:PC - Parametric Constructor triggered via StorageInterface.\n";
}
// --- Public Interface ---
void save(std::ostream& os) const
{
if(pimpl_) pimpl_->serialize(os);
}
};
//--------------------------------------------------------- 3. Main simulation:
int main()
{
std::cout << "=== TYPE ERASURE: NON-INTRUSIVE SERIALIZATION (C++20) ===\n" << std::endl;
std::vector<SerializableEntity> archive;
std::cout << "--- PHASE 1: Adding objects to the archive ---\n";
archive.push_back(User{"Mario"});
archive.push_back(NetworkConfig{8080});
archive.push_back(User{"Bjarne"});
std::cout << "\n--- PHASE 2: Demonstrating Deep Copy (Rule 2:CC) ---\n";
auto archive_backup = archive;
std::cout << "\n--- PHASE 3: Processing the Archive ---\n";
for(const auto& entity : archive)
{
std::cout << " -> Serializing: ";
entity.save(std::cout);
std::cout << "\n";
}
std::cout << "\n--- PHASE 4: Processing the Backup ---\n";
for(const auto& entity : archive_backup)
{
std::cout << " -> Backup Data: ";
entity.save(std::cout);
std::cout << "\n";
}
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: TaskQueue.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program demonstrates Type Erasure applied to an execution pipeline.
* We create an 'AnyTaskEntity' wrapper that can store and execute any "Callable"
* object (lambdas, functors, or function pointers) without a shared
* inheritance hierarchy.
*
* --- C++20 CONCEPTS:
* We define a 'Callable' concept to ensure that only types providing
* an 'operator()' can be wrapped as a task.
*
* --- THE RULE OF SEVEN:
* 'AnyTaskEntity' manages its internal 'StorageInterface' via unique_ptr,
* implementing the full lifecycle to allow tasks to be stored in
* standard containers, moved between queues, or duplicated.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>
#include <concepts>
//--------------------------------------------------------- 0. Formal Contract:
// A type is Callable if it can be invoked with no arguments.
template <class T>
concept Callable = requires(T& object)
{
object();
};
//--------------------------------------------------------- 1. Type erasure container:
class AnyTaskEntity
{
private:
// --- Internal Infrastructure ---
class StorageInterface
{
public:
virtual ~StorageInterface() = default;
virtual void execute()                                  = 0;
virtual std::unique_ptr<StorageInterface> clone() const = 0;
};
// Only accepts types that satisfy the 'Callable' concept.
template <Callable TaskType>
class Model final : public StorageInterface
{
private:
TaskType function_;
public:
explicit Model(TaskType&& value) : function_{std::move(value)} { }
explicit Model(const TaskType& value) : function_{value} { }
// --- Domain Operation (Internal/Contractual) ---
// Delegates directly to the business class. The wrapped type must natively 
// implement this method to satisfy the compile-time 'Serializable' concept.
void execute() override
{
function_();
}
// --- Extraneous Extension (External/Non-Intrusive) ---
// This is an external capability injected solely by the Type Erasure wrapper.
// The business classes are completely unaware of 'clone()', demonstrating how 
// this pattern allows adding new polymorphic behaviors without altering the original classes.
std::unique_ptr<StorageInterface> clone() const override
{
return std::unique_ptr< Model<TaskType> >(new Model<TaskType>(function_));
}
};
std::unique_ptr<StorageInterface> pimpl_;
public:
// --- APPLYING THE RULE OF SEVEN ---
// 1:DC -  Default Constructor: Disabled.
AnyTaskEntity() = delete;
// 2:CC - Copy Constructor: Deep copy for duplicating tasks
AnyTaskEntity(const AnyTaskEntity& other)
: pimpl_{other.pimpl_ ? other.pimpl_->clone() : nullptr}
{
std::cout << " [Rule of Seven] 2:CC - Deep Copy Constructor.\n";
}
// 3:MC - Move Constructor
AnyTaskEntity(AnyTaskEntity&& other) noexcept : pimpl_{std::move(other.pimpl_)}
{
std::cout << " [Rule of Seven] 3:MC - Move Constructor.\n";
}
// 4:CA - Copy Assignment
AnyTaskEntity& operator=(const AnyTaskEntity& other)
{
std::cout << " [Rule of Seven] 4:CA - Copy Assignment.\n";
if(this != &other)
{
AnyTaskEntity temp(other);
std::swap(pimpl_, temp.pimpl_);
}
return *this;
}
// 5:MA - Move Assignment
AnyTaskEntity& operator=(AnyTaskEntity&& other) noexcept
{
std::cout << " [Rule of Seven] 5:MA - Move Assignment.\n";
if(this != &other) pimpl_ = std::move(other.pimpl_);
return *this;
}
// 6:De - Destructor
~AnyTaskEntity()
{
if(pimpl_) std::cout << " [Rule of Seven] 6:De - Destructor - Task released.\n";
}
// 7:PC - Parametric Constructor (Template): Captures any Callable type.
// Only accepts types that satisfy the 'Callable' concept.
template <Callable TaskType>
AnyTaskEntity(TaskType&& object)
: pimpl_{std::make_unique< Model<std::decay_t<TaskType>> >(std::forward<TaskType>(object))}
{
std::cout << " [Rule of Seven] 7:PC - Parametric Constructor - Task captured via StorageInterface.\n";
}
// --- Public Interface ---
void execute()
{
if(pimpl_) pimpl_->execute();
}
};
//--------------------------------------------------- 2. Custom Functor (Task):
// Example of a class that is NOT designed for this system but is compatible.
class LogWorker
{
private:
std::string id_;
public:
explicit LogWorker(std::string id) : id_{std::move(id)} { }
void operator()() { std::cout << " -> [LogWorker " << id_ << "] is processing logs.\n"; }
};
//--------------------------------------------------------- 3. Main simulation:
int main()
{
std::cout << "=== TYPE ERASURE: ASYNCHRONOUS TASK QUEUE ===\n" << std::endl;
std::vector<AnyTaskEntity> taskQueue;
std::cout << "--- PHASE 1: Loading heterogeneous tasks ---\n";
// Task A: A lambda with capture
std::string secret = "Data_42";
taskQueue.push_back([secret]()
{
std::cout << " -> [Lambda] Accessing captured state: " << secret << "\n";
});
// Task B: A custom functor object
taskQueue.push_back(LogWorker{"Alpha"});
// Task C: Another lambda
taskQueue.push_back([]()
{
std::cout << " -> [Lambda] Performing a quick calculation: " << 10 + 20 << "\n";
});
std::cout << "\n--- PHASE 2: Duplicating the Queue (Deep Copy) ---\n";
auto backupQueue = taskQueue;
std::cout << "\n--- PHASE 3: Executing the Primary Queue ---\n";
for(auto& task : taskQueue) task.execute();
std::cout << "\n--- PHASE 4: Executing the Backup Queue ---\n";
for(auto& task : backupQueue) task.execute();
std::cout << "\n=== SIMULATION COMPLETED ===\n";
}
//================================================================================ END
/**
* ============================================================================
* File: Traditional.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program demonstrates the traditional approach to operator overloading
* for mathematical vectors, implemented using modern C++ standards. It
* represents a robust, professional-grade implementation that fully utilizes
* Move Semantics to optimize memory management.
*
* --- ARCHITECTURAL ANALYSIS:
* Although this implementation avoids unnecessary deep copies by recycling
* temporary buffers (Move Semantics), it remains fundamentally limited by the
* "Memory Wall". In a complex expression like R = A + 2.0*B + 3.0*C:
* 1. Sequential Execution: Each operator (+, *) triggers its own independent
*    loop, forcing the CPU to traverse the data multiple times.
* 2. Memory Pressure: Intermediate temporary results still require full-sized
*    buffers (~8.94 GiB each). At the peak of evaluation, the total memory
*    requirement exceeds physical RAM, triggering OS Swapping and increasing
*    latency.
*
* --- PERFORMANCE NOTES:
* This baseline profiles the maximum throughput achievable through standard
* polymorphism and buffer recycling before introducing Loop Fusion.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <utility>
// CALIBRATION CONSTANTS:
// Try to get approximately 5 seconds of execution time or use up all your memory.
// 1.2 billion elements require ~8.94 GB of RAM per vector.
const size_t VECTOR_SIZE = 1'200'000'000;
// Calibrate WARMUP_ITERATIONS to achieve approximately 5 seconds of
// execution time to stabilize CPU frequency and cache lines.
const size_t WARMUP_ITERATIONS = 3'000'000'000;
//--------------------------------------------------------- Traditional Vector:
class Vector
{
private:
std::vector<double> data_;
public:
// Constructors
explicit Vector(size_t size) : data_(size) { }
Vector(size_t size, double value) : data_(size, value) { }
size_t size() const { return data_.size(); }
double operator[](size_t i) const { return data_[i]; }
double& operator[](size_t i) { return data_[i]; }
// --- Addition Operators (Overloaded to handle all Value Categories) ---
// Lvalue + Lvalue (Construct a new Vector)
friend Vector operator+(const Vector& lhs, const Vector& rhs)
{
Vector result(lhs.size());
for(size_t i = 0; i < lhs.size(); ++i) result[i] = lhs[i] + rhs[i];
return result;
// Copy Elision - NRVO (Named Return Value Optimization)
}
// Rvalue + Lvalue (Recycle LHS)
friend Vector operator+(Vector&& lhs, const Vector& rhs)
{
for(size_t i = 0; i < lhs.size(); ++i) lhs.data_[i] += rhs.data_[i];
return std::move(lhs);
}
// Lvalue + Rvalue (Recycle RHS)
friend Vector operator+(const Vector& lhs, Vector&& rhs)
{
for(size_t i = 0; i < rhs.size(); ++i) rhs.data_[i] += lhs.data_[i];
return std::move(rhs);
}
// Rvalue + Rvalue (Recycle LHS - Release RHS)
friend Vector operator+(Vector&& lhs, Vector&& rhs)
{
for(size_t i = 0; i < lhs.size(); ++i) lhs.data_[i] += rhs.data_[i];
rhs = Vector(0);
// Release rhs memory using default move assigment
return std::move(lhs);
}
// --- Multiplication Operators ---
// Factor * Lvalue (Construct a new Vector)
friend Vector operator*(double factor, const Vector& vec)
{
Vector result(vec.size());
for(size_t i = 0; i < vec.size(); ++i) result[i] = factor * vec[i];
return result;
// Copy Elision - NRVO (Named Return Value Optimization)
}
// Factor * Rvalue (Recycle Vector)
friend Vector operator*(double factor, Vector&& vec)
{
for(size_t i = 0; i < vec.size(); ++i) vec.data_[i] *= factor;
return std::move(vec);
}
};
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== EXPRESSION TEMPLATES: TRADITIONAL BASELINE ===\n" << std::endl;
// 1. DATA INITIALIZATION
std::cout << " [1/4] Initializing vectors of size: " << VECTOR_SIZE << "...\n";
Vector A(VECTOR_SIZE, 1.0);
Vector B(VECTOR_SIZE, 2.0);
Vector C(VECTOR_SIZE, 3.0);
Vector R(VECTOR_SIZE);
// 2. CPU WARM-UP PHASE
std::cout << " [2/4] Warming up CPU (Target: ~5 seconds)..." << std::endl;
volatile double warm = 0.0;
for(size_t i = 0; i < WARMUP_ITERATIONS; ++i)
warm += std::sqrt(static_cast<double>(i));
// 3. BENCHMARK MEASUREMENT
std::cout << " [3/4] Executing: R = 2.0 * ( A + 3.0 * B + 4.0 * C ) ..." << std::endl;
auto start = std::chrono::high_resolution_clock::now();
// This operation triggers the 'Memory Wall' bottleneck.
// Peak memory usage will reach its maximum during the evaluation of sub-expressions.
R = 2.0 * ( A + 3.0 * B + 4.0 * C );
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
// 4. RESULTS REPORT AND VERIFICATION
std::cout << " [4/4] Verification - R[0]: " << R[0] << " (Expected: 38)\n";
std::cout << "\n Elapsed time: " << elapsed.count() << " seconds.\n";
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Expressions_CRTP.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program demonstrates the "Expression Templates" pattern using the
* Curiously Recurring Template Pattern (CRTP). It solves the massive memory
* and speed bottlenecks of the traditional approach by using "Lazy Evaluation"
* and "Loop Fusion".
*
* --- CRTP MECHANICS:
* The base class 'VecExpression<Derived>' is a template that receives the 
* concrete type of its child. Methods like 'size()' and 'operator[]' 
* utilize 'static_cast<const Derived&>(*this)' to delegate the call 
* to the specific implementation (Vector, VecSum, or VecScale) at 
* compile-time, bypassing the virtual table.
*
* --- THE ARCHITECTURAL MAGIC:
* 1. Expression Proxy: Operators (+, *) no longer perform immediate calculations.
*    Instead, they return lightweight proxy objects (representing the Abstract
*    Syntax Tree) at compile-time with zero allocations.
* 2. Loop Fusion: The entire mathematical expression is fused into a single,
*    contiguous 'for' loop inside the assignment operator of the 'Vector' class.
* 3. Zero-Allocation: No intermediate temporary Vector objects (like 8.94 GB
*    buffers) are created on the heap during the evaluation of the expression.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
// CALIBRATION CONSTANTS (Identical to Traditional.cpp for a fair benchmark)
const size_t VECTOR_SIZE       = 1'200'000'000;
const size_t WARMUP_ITERATIONS = 3'000'000'000;
//--------------------------------------------------------- 1. CRTP Base Class:
// This empty interface forces compile-time polymorphism without virtual tables.
// It ensures that all expressions share a common structural contract.
template <class Derived>
class VecExpression
{
public:
// Static Polymorphism: Derived can be a Vector, a VecSum or a VecScale.
// CRTP requires an explicit static_cast to access the derived members.
auto size() const { return static_cast<const Derived&>(*this).size(); }
auto operator[](size_t i) const { return static_cast<const Derived&>(*this)[i]; }
};
//--------------------------------------------------------- 2. Vector Container:
class Vector : public VecExpression<Vector>
{
private:
std::vector<double> data_;
public:
// Constructors
explicit Vector(size_t size) : data_(size) { }
Vector(size_t size, double value) : data_(size, value) { }
size_t size() const { return data_.size(); }
double operator[](size_t i) const { return data_[i]; }
//double& operator[](size_t i) { return data_[i]; }
// Lazy Evaluation Assignment Operator
// This is where the "Loop Fusion" occurs.
// It accepts any expression node and evaluates it in a single pass.
template <class Expression>
Vector& operator=(const Expression& expr)
{
for(size_t i = 0; i < expr.size(); ++i)
data_[i] = expr[i];
// data_[i] = 2.0 * ( A[i] + (3.0 * B[i]) + (4.0 * C[i]) )
return *this;
}
};
//--------------------------------------------------------- 3. VecSum Node:
// Represents a delayed sum of two independent expressions.
template <class LHS_Expr, class RHS_Expr>
class VecSum : public VecExpression<VecSum<LHS_Expr, RHS_Expr>>
{
private:
const LHS_Expr& lhs_;
// Reference to the left-hand side expression operand
const RHS_Expr& rhs_;
// Reference to the right-hand side expression operand
public:
VecSum(const LHS_Expr& lhs, const RHS_Expr& rhs) : lhs_(lhs), rhs_(rhs) { }
size_t size() const { return lhs_.size(); }
// Inline element access: propagates index requests down the expression tree
double operator[](size_t i) const { return lhs_[i] + rhs_[i]; }
};
//--------------------------------------------------------- 4. VecScale Node:
// Represents a delayed multiplication of a scalar value and an expression.
template <class Expression>
class VecScale : public VecExpression<VecScale<Expression>>
{
private:
double            factor_;
// The scaling factor
const Expression& expr_;
// Reference to the expression being scaled
public:
VecScale(double factor, const Expression& expr) : factor_(factor), expr_(expr) { }
size_t size() const { return expr_.size(); }
// Inline element access: performs scalar multiplication on the fly
double operator[](size_t i) const { return factor_ * expr_[i]; }
};
//--------------------------------------------------------- 5. Operator Overloads:
// These operators DO NOT execute any loops or allocate heap memory.
// They simply deduce and assemble the Abstract Syntax Tree (AST) at compile-time.
// Non-member operator+ for two arbitrary expressions
template <class LHS_Expr, class RHS_Expr>
auto operator+(const LHS_Expr& lhs, const RHS_Expr& rhs)
{
return VecSum<LHS_Expr, RHS_Expr>(lhs, rhs);
}
// Non-member operator* for a scalar factor multiplying an arbitrary expression
template <class Expression>
auto operator*(double factor, const Expression& expr)
{
return VecScale<Expression>(factor, expr);
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== EXPRESSION TEMPLATES: CRTP & LOOP FUSION ===\n" << std::endl;
// 1. DATA INITIALIZATION (Identical sequence to traditional version)
std::cout << " [1/4] Initializing vectors of size: " << VECTOR_SIZE << "...\n";
Vector A(VECTOR_SIZE, 1.0);
Vector B(VECTOR_SIZE, 2.0);
Vector C(VECTOR_SIZE, 3.0);
Vector R(VECTOR_SIZE);
// 2. CPU WARM-UP PHASE
std::cout << " [2/4] Warming up CPU (Target: ~5 seconds)..." << std::endl;
volatile double warm = 0.0;
for(size_t i = 0; i < WARMUP_ITERATIONS; ++i)
warm += std::sqrt(static_cast<double>(i));
// 3. BENCHMARK MEASUREMENT
std::cout << " [3/4] Executing: R = 2.0 * ( A + 3.0 * B + 4.0 * C ) ..." << std::endl;
auto start = std::chrono::high_resolution_clock::now();
/**
* THE EXPRESSION TEMPLATE MAGIC:
*
* Unlike the traditional approach, no temporary vectors are created.
* The compiler builds a static AST and executes a single loop.
*
* Visually, the compiled static AST looks like this:
*
*                VecScale(2*(A + 3*B + 4*C))
*                   |
*                VecSum(A + 3*B + 4*C)
*               /       \
*         VecSum(A+3*B)  VecScale(4*C)
*        /      \
*    Vector(A)   VecScale(3*B)
*
* When assigned to R, the operator= triggers a single, fused, highly
* optimized loop: R[i] = 2.0 * ( A[i] + (3.0 * B[i]) + (4.0 * C[i]) )
*
* Zero temporary vectors are allocated on the heap during evaluation.
*/
R = 2.0 * ( A + 3.0 * B + 4.0 * C );
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
// 4. RESULTS REPORT AND VERIFICATION
std::cout << " [4/4] Verification - R[0]: " << R[0] << " (Expected: 38)\n";
std::cout << "\n Elapsed time: " << elapsed.count() << " seconds.\n";
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
/**
* ============================================================================
* File: Expressions_Deducing_this.cpp
* Author: Mario Galindo Queralt, Ph.D.
*
* --- DESIGN OVERVIEW:
* This program implements the "Expression Templates" pattern using the
* modern C++23 "Deducing This" feature (Explicit Object Parameters). It
* represents the state-of-the-art evolution of high-performance mathematical
* DSLs (Domain Specific Languages) in C++.
*
* --- THE UNIFIED INTERFACE ADVANTAGE:
* Unlike more complex implementations where operators (+, *) must be
* redefined inside every node type (Vector, VecSum, VecScale), we introduce
* a non-template base class: 'VecExpression'.
*
* By having all expression nodes inherit from this single interface:
* 1. Code Reusability (DRY): Mathematical operators are defined once as
*    global templates. Because they target 'VecExpression', they
*    automatically work for any current or future node in the tree.
* 2. Static Contract: It enforces a uniform interface (size and operator[])
*    across the entire hierarchy without the overhead of virtual functions.
*
* --- DEDUCING THIS (C++23) vs. CRTP:
* This version eliminates the "Curiously Recurring" template syntax. By using
* 'this auto&& self' in the base class methods, the compiler automatically
* deduces the concrete derived type (Vector, VecSum, etc.) at the call site.
* This removes the need for 'static_cast' and makes the code significantly
* more readable while maintaining identical zero-overhead performance.
*
* --- PERFORMANCE & LOOP FUSION:
* The traditional "Naive" approach creates costly temporary vectors for
* every sub-expression. This implementation:
* 1. Eliminates Temporaries: Operations return lightweight proxies that
*    act as an Abstract Syntax Tree (AST).
* 2. Single-Pass Execution: The entire calculation is collapsed into one
*    continuous loop inside the assignment operator, maximizing CPU cache
*    locality and overcoming the "Memory Wall" bottleneck.
* ============================================================================
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
// CALIBRATION CONSTANTS (Identical to Traditional.cpp for a fair benchmark)
const size_t VECTOR_SIZE       = 1'200'000'000;
const size_t WARMUP_ITERATIONS = 3'000'000'000;
//--------------------------------------------------------- 1. Base Class:
// C++23 Base: Simple, readable, and non-template.
class VecExpression
{
public:
// 'this auto&& self' deduces if we are a Vector, a VecSum, or a VecScale.
// No static_cast required.
auto size(this auto&& self) { return self.size(); }
auto operator[](this auto&& self, size_t i) { return self[i]; }
};
//--------------------------------------------------------- 2. Vector Container:
class Vector : public VecExpression
{
private:
std::vector<double> data_;
public:
// Constructors
explicit Vector(size_t size) : data_(size) { }
Vector(size_t size, double value) : data_(size, value) { }
size_t size() const { return data_.size(); }
double operator[](size_t i) const { return data_[i]; }
//double& operator[](size_t i) { return data_[i]; }
// Lazy Evaluation Assignment Operator
// This is where the "Loop Fusion" occurs.
// It accepts any expression node and evaluates it in a single pass.
template <class Expression>
Vector& operator=(const Expression& expr)
{
for(size_t i = 0; i < expr.size(); ++i)
data_[i] = expr[i];
// data_[i] = 2.0 * ( A[i] + (3.0 * B[i]) + (4.0 * C[i]) )
return *this;
}
};
//--------------------------------------------------------- 3. VecSum Node:
// Represents a delayed sum of two independent expressions.
template <class LHS_Expr, class RHS_Expr>
class VecSum : public VecExpression
{
private:
const LHS_Expr& lhs_;
// Reference to the left-hand side expression operand
const RHS_Expr& rhs_;
// Reference to the right-hand side expression operand
public:
VecSum(const LHS_Expr& lhs, const RHS_Expr& rhs) : lhs_(lhs), rhs_(rhs) { }
size_t size() const { return lhs_.size(); }
// Inline element access: propagates index requests down the expression tree
double operator[](size_t i) const { return lhs_[i] + rhs_[i]; }
};
//--------------------------------------------------------- 4. VecScale Node:
// Represents a delayed multiplication of a scalar value and an expression.
template <class Expression>
class VecScale : public VecExpression
{
private:
double            factor_;
// The scaling factor
const Expression& expr_;
// Reference to the expression being scaled
public:
VecScale(double factor, const Expression& expr) : factor_(factor), expr_(expr) { }
size_t size() const { return expr_.size(); }
// Inline element access: performs scalar multiplication on the fly
double operator[](size_t i) const { return factor_ * expr_[i]; }
};
//--------------------------------------------------------- 5. Operator Overloads:
// These operators DO NOT execute any loops or allocate heap memory.
// They simply deduce and assemble the Abstract Syntax Tree (AST) at compile-time.
// Non-member operator+ for two arbitrary expressions
template <class LHS_Expr, class RHS_Expr>
auto operator+(const LHS_Expr& lhs, const RHS_Expr& rhs)
{
return VecSum<LHS_Expr, RHS_Expr>(lhs, rhs);
}
// Non-member operator* for a scalar factor multiplying an arbitrary expression
template <class Expression>
auto operator*(double factor, const Expression& expr)
{
return VecScale<Expression>(factor, expr);
}
//--------------------------------------------------------- Main Simulation:
int main()
{
std::cout << "=== EXPRESSION TEMPLATES: DEDUCING THIS (C++23) ===\n" << std::endl;
// 1. DATA INITIALIZATION (Identical sequence to traditional version)
std::cout << " [1/4] Initializing vectors of size: " << VECTOR_SIZE << "...\n";
Vector A(VECTOR_SIZE, 1.0);
Vector B(VECTOR_SIZE, 2.0);
Vector C(VECTOR_SIZE, 3.0);
Vector R(VECTOR_SIZE);
// 2. CPU WARM-UP PHASE
std::cout << " [2/4] Warming up CPU (Target: ~5 seconds)..." << std::endl;
volatile double warm = 0.0;
for(size_t i = 0; i < WARMUP_ITERATIONS; ++i)
warm += std::sqrt(static_cast<double>(i));
// 3. BENCHMARK MEASUREMENT
std::cout << " [3/4] Executing: R = 2.0 * ( A + 3.0 * B + 4.0 * C ) ..." << std::endl;
auto start = std::chrono::high_resolution_clock::now();
/**
* THE EXPRESSION TEMPLATE MAGIC:
*
* Unlike the traditional approach, no temporary vectors are created.
* The compiler builds a static AST and executes a single loop.
*
* Visually, the compiled static AST looks like this:
*
*                VecScale(2*(A + 3*B + 4*C))
*                   |
*                VecSum(A + 3*B + 4*C)
*               /       \
*         VecSum(A+3*B)  VecScale(4*C)
*        /      \
*    Vector(A)   VecScale(3*B)
*
* When assigned to R, the operator= triggers a single, fused, highly
* optimized loop: R[i] = 2.0 * ( A[i] + (3.0 * B[i]) + (4.0 * C[i]) )
*
* Zero temporary vectors are allocated on the heap during evaluation.
*/
R = 2.0 * ( A + 3.0 * B + 4.0 * C );
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
// 4. RESULTS REPORT AND VERIFICATION
std::cout << " [4/4] Verification - R[0]: " << R[0] << " (Expected: 38)\n";
std::cout << "\n Elapsed time: " << elapsed.count() << " seconds.\n";
std::cout << "\n=== SIMULATION COMPLETED ===" << std::endl;
}
//================================================================================ END
