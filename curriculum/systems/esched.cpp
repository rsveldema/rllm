#include "ChannelsTest.hpp"
#include <thread>
#include <array>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
using namespace std::chrono_literals;
namespace{
Timeout testtime;
}
template<typename T>
class IChannel{
public:
virtual void send(const T& t) = 0;
virtual T recv() = 0;
};
template<typename T>
class BufferChannel : public IChannel<T>{
private:
uint32_t rp = 0;
uint32_t wp = 0;
std::array<T, 16> data;
std::mutex m;
std::condition_variable cv;
public:
/** Nonblocking: overwrites old elements if full
*/
void send(const T& t) override{
std::scoped_lock l(m);
data[wp] = t;
wp = (wp + 1) % data.size();
if (wp == rp){
rp = (rp + 1) % data.size();
}
cv.notify_one();
}
/** blocking if channel is empty
*/
T recv() override{
std::unique_lock<std::mutex> lk(m);
cv.wait(lk, [this]() { return rp != wp; });
auto ret = data[rp];
rp = (rp + 1) % data.size();
return ret;
}
};
class PeriodicChannel : IChannel<std::chrono::milliseconds>{
private:
std::chrono::milliseconds period;
public:
PeriodicChannel(const std::chrono::milliseconds _period)
: period(_period){
}
void send(const std::chrono::milliseconds& t) override{
throw new std::exception();
// unimpl
}
std::chrono::milliseconds recv() override{
std::this_thread::sleep_for(period);
return period;
}
};
template<typename T>
class SensorChannel : public IChannel<T>{
private:
ISensor<T>& sensor;
public:
SensorChannel(ISensor<T>& _sensor) : sensor(_sensor) {}
void send(const T& t) override{
throw new std::exception();
// unimplemented
}
T recv() override{
return sensor.GetValue();
}
};
class LEDChannel : public IChannel<bool>{
private:
ILED& led;
public:
LEDChannel(ILED& _led) : led(_led) {}
void send(const bool& value) override{
if (value){
led.on();
}
else {
led.off();
}
}
bool recv() override{
throw new std::exception();
// unimplemented
}
};
class UploadChannel : public IChannel<std::string>{
private:
INetworkDevice& net;
std::unique_ptr<IConnection> conn;
std::shared_ptr<ISocket> socket;
public:
UploadChannel(INetworkDevice& _net) : net(_net) {}
void send(const std::string& value) override{
if (!conn){
conn = std::move(net.CreateConnectionRequest("example.com", 80, Protocol::SSL));
while (! conn->PollSocketReady().has_value()){
}
socket = conn->PollSocketReady().value();
}
socket->Write("msg: " + value);
}
std::string recv() override{
throw new std::exception();
// unimplemented
}
};
void channelsTest(const HAL& hal){
testtime = Timeout(30s);
auto periodic1s_led = PeriodicChannel(1s);
auto ledCh = LEDChannel(hal.led);
auto periodic1s = PeriodicChannel(1s);
auto sensorCh = SensorChannel(hal.sensor);
auto periodic5s = PeriodicChannel(5s);
auto uploadCh = UploadChannel(hal.net);
auto bufferCh = BufferChannel<int>();
std::thread ledTask([&]() {
while (!testtime.elapsed()){
periodic1s_led.recv();
ledCh.send(true);
periodic1s_led.recv();
ledCh.send(false);
}
});
std::thread sensorTask([&]() {
while (!testtime.elapsed()){
periodic1s.recv();
bufferCh.send(sensorCh.recv());
}
});
std::thread uploadTask([&]() {
while (!testtime.elapsed()){
std::string msg;
for (int i = 0; i < 5; i++){
msg += " " + std::to_string(bufferCh.recv());
}
uploadCh.send(msg);
}
});
ledTask.join();
sensorTask.join();
uploadTask.join();
}
#include "HAL.hpp"
#include <iostream>
#include <signal.h>
#include <assert.h>
#ifdef __linux
#include <sys/time.h>
#else
#endif
class MockedLED : public ILED{
public:
void on() override { std::cerr << "on\n"; }
void off() override { std::cerr << "off\n"; }
};
class MockWriteCompletion : public ICompletion{
public:
MockWriteCompletion(const std::string &msg){
}
std::optional<CompletionResult> finished() override{
if (counter++ > 3){
std::cerr << "socket-written !!!!\n";
return CompletionResult{};
}
return std::nullopt;
}
private:
int counter = 0;
};
class MockReadCompletion : public ICompletion{
public:
MockReadCompletion(const Timeout &timeout){
}
std::optional<CompletionResult> finished() override{
if (counter++ > 3){
std::cerr << "socket-recv\n";
return CompletionResult{};
}
return std::nullopt;
}
private:
int counter = 0;
};
class MockSocket : public ISocket{
public:
virtual void Write(const std::string &msg){
std::cerr << "socket-send: " << std::to_string(msg.size()) << "\n";
}
virtual std::string Read(Timeout timeout){
std::string msg("reply-msg");
std::cerr << "socket-recv: " << std::to_string(msg.size()) << "\n";
return msg;
}
std::shared_ptr<ICompletion> WriteAsync(const std::string &msg){
return std::make_shared<MockWriteCompletion>(msg);
}
std::shared_ptr<ICompletion> ReadAsync(Timeout timeout){
return std::make_shared<MockReadCompletion>(timeout);
}
};
class DummyConnectionRequest : public IConnection{
public:
DummyConnectionRequest(const std::string &host, int port, Protocol protocol){
}
// Inherited via IConnection
virtual std::optional<std::shared_ptr<ISocket>> PollSocketReady() override{
auto ret = std::make_shared<MockSocket>();
return std::make_optional<std::shared_ptr<ISocket>>(ret);
}
};
class NetworkDevice : public INetworkDevice{
public:
std::unique_ptr<IConnection> CreateConnectionRequest(const std::string &host, int port, Protocol protocol) override{
return std::make_unique<DummyConnectionRequest>(host, port, protocol);
}
};
class Sensor : public ISensor<int8_t>{
public:
// Inherited via ISensor
virtual int8_t GetValue() override{
static int counter = 0;
std::cerr << "sensor value: " << std::to_string(counter) << "\n";
return counter++;
}
};
HAL CreateHAL(){
static MockedLED mockedLED;
static NetworkDevice mockedNetwork;
static Sensor mockedSensor;
return {mockedLED, mockedNetwork, mockedSensor};
}
#ifdef WIN32
#include <Windows.h>
#include <WinUser.h>
static void (*globalTimerInterruptFunc)();
VOID timerfunc(HWND, UINT, UINT_PTR, DWORD){
if (globalTimerInterruptFunc){
globalTimerInterruptFunc();
}
}
void HAL::SetTimerInterrupt(std::chrono::milliseconds ms, void (*func)()) const{
globalTimerInterruptFunc = func;
static UINT_PTR uIDEvent;
if (func){
uIDEvent = SetTimer(nullptr, 0, (UINT)ms.count(), timerfunc);
assert(uIDEvent != 0);
}
else{
KillTimer(nullptr, uIDEvent);
}
}
void HAL::WaitForInterupt() const{
MSG msg;
// message structure
if (GetMessage(&msg,
// message structure
NULL,
// handle to window to receive the message
0,
// lowest message to examine
0))
// highest message to examine{
TranslateMessage(&msg);
// translates virtual-key codes
DispatchMessage(&msg);
// dispatches message to window
}
}
#else
static void (*globalTimerInterruptFunc)();
void trampoline_alarm(int arg){
if (globalTimerInterruptFunc){
globalTimerInterruptFunc();
}
}
void HAL::SetTimerInterrupt(std::chrono::milliseconds ms, void (*func)()) const{
globalTimerInterruptFunc = func;
if (func == nullptr){
signal(SIGALRM, SIG_DFL);
int ret = setitimer(ITIMER_REAL, nullptr, nullptr);
assert(ret == 0);
return;
}
signal(SIGALRM, trampoline_alarm);
timeval tv;
tv.tv_sec = ms.count() / 1000;
tv.tv_usec = (ms.count() % 1000) * 1000;
itimerval val;
val.it_interval = tv;
val.it_value = tv;
int ret = setitimer(ITIMER_REAL, &val, nullptr);
assert(ret == 0);
}
void HAL::WaitForInterupt() const{
}
#endif#include <memory>
#include <iostream>
#include <vector>
#include <queue>
#include <assert.h>
#include <functional>
#include "CoroutineTest.hpp"
#include "Timeout.hpp"
#include <optional>
#ifndef NO_CO_ROUTINE_SUPPORT
#include "await_support.hpp"
using namespace std::chrono_literals;
namespace {
Timeout testtime;
using ITask = Task<int>;
ITask::AWaitConditionProxy coroutine_sleep(std::chrono::milliseconds ms){
Timeout timeout(ms);
return ITask::AWaitConditionProxy(std::make_shared<LambdaCondition>([timeout]() {
return timeout.elapsed();
}));
}
ITask::AWaitConditionProxy coroutine_have_socket(std::shared_ptr<IConnection> conn){
return ITask::AWaitConditionProxy(std::make_shared<LambdaCondition>([conn]() {
return conn->PollSocketReady() != nullptr;
}));
}
ITask::AWaitConditionProxy coroutine_completion_finished(std::shared_ptr<ICompletion> completion){
return ITask::AWaitConditionProxy(std::make_shared<LambdaCondition>([completion]() {
return completion->finished().has_value();
}));
}
ITask blinkerTask(ILED &led){
while (true){
led.on();
co_await coroutine_sleep(1s);
led.off();
co_await coroutine_sleep(1s);
}
co_return 0;
}
std::queue<int8_t> sensorValues;
ITask sensorTask(ISensor<int8_t> &sensor){
while (true){
auto value = sensor.GetValue();
sensorValues.push(value);
co_await coroutine_sleep(1s);
}
co_return 0;
}
ITask networkTask(INetworkDevice &net){
auto uniq = std::move(net.CreateConnectionRequest("example.com", 80, Protocol::TCP));
std::shared_ptr<IConnection> conn = std::move(uniq);
co_await coroutine_have_socket(conn);
auto socket_opt = conn->PollSocketReady();
assert(socket_opt);
auto socket = socket_opt.value();
while (true){
co_await coroutine_sleep(5s);
std::vector<int8_t> msg;
while (!sensorValues.empty()){
msg.push_back(sensorValues.front());
sensorValues.pop();
}
auto completion = socket->WriteAsync(std::string("payload: ") + std::to_string(msg.size()) + ":" + std::string((char *)msg.data(), msg.size()));
co_await coroutine_completion_finished(completion);
}
co_return 0;
}
}
// namespace
void coroutineTest(const HAL &hal){
testtime = Timeout(30s);
auto btask = blinkerTask(hal.led);
auto stask = sensorTask(hal.sensor);
auto ntask = networkTask(hal.net);
while (!testtime.elapsed()){
btask.try_schedule();
stask.try_schedule();
ntask.try_schedule();
}
}
#else
void coroutineTest(const HAL &hal)
{}
#endif
#include "functionalTest.hpp"
#include <functional>
#include <memory>
#include <queue>
using namespace std::chrono_literals;
/**
* The idea is to, whenever you need to block/wait, return/pass a function to continue execution whenever that condition has been fulfilled.
*/
namespace{
Timeout testtime;
class Func{
public:
Func(const HAL &hal) : hal(hal) {}
virtual std::shared_ptr<Func> run() = 0;
protected:
const HAL &hal;
};
struct WaitFunc : Func{
WaitFunc(const HAL &hal, std::function<bool()> cond, std::shared_ptr<Func> onFinish) : Func(hal), cond(cond), onFinish(onFinish) {}
std::shared_ptr<Func> run() override{
if (cond()){
return onFinish;
}
return std::make_shared<WaitFunc>(hal, cond, onFinish);
}
private:
std::function<bool()> cond;
std::shared_ptr<Func> onFinish;
};
struct CombineFunc : Func{
CombineFunc(const HAL &hal, std::shared_ptr<Func> f1, std::shared_ptr<Func> f2) : Func(hal), f1(f1), f2(f2) {}
std::shared_ptr<Func> run() override{
f1 = f1->run();
f2 = f2->run();
return std::make_shared<CombineFunc>(hal, f1, f2);
}
private:
std::shared_ptr<Func> f1;
std::shared_ptr<Func> f2;
};
struct BlinkOff : Func{
BlinkOff(const HAL &hal) : Func(hal) {}
std::shared_ptr<Func> run() override;
};
struct BlinkOn : Func{
BlinkOn(const HAL &hal) : Func(hal) {}
std::shared_ptr<Func> run() override;
};
std::shared_ptr<Func> BlinkOn::run(){
hal.led.on();
Timeout timeout(1s);
return std::make_shared<WaitFunc>(
hal, [timeout]() { return timeout.elapsed(); }, std::make_shared<BlinkOff>(hal));
}
std::shared_ptr<Func> BlinkOff::run(){
hal.led.off();
Timeout timeout(1s);
return std::make_shared<WaitFunc>(
hal, [timeout]() { return timeout.elapsed(); }, std::make_shared<BlinkOn>(hal));
}
std::queue<int8_t> sensorValues;
struct ReadSensor : Func{
ReadSensor(const HAL &hal) : Func(hal) {}
std::shared_ptr<Func> run() override{
auto value = hal.sensor.GetValue();
sensorValues.push(value);
Timeout timeout(1s);
return std::make_shared<WaitFunc>(
hal, [timeout]() { return timeout.elapsed(); }, std::make_shared<ReadSensor>(hal));
}
};
struct SendViaSocket : Func{
SendViaSocket(std::shared_ptr<ISocket> &socket, const HAL &hal) : Func(hal), socket(socket){
}
std::shared_ptr<Func> run() override{
std::vector<int8_t> msg;
while (!sensorValues.empty()){
msg.push_back(sensorValues.front());
sensorValues.pop();
}
socket->Write(std::string("payload: ") + std::to_string(msg.size()) + ":" + std::string((char *)msg.data(), msg.size()));
Timeout timeout(5s);
return std::make_shared<WaitFunc>(
hal, [timeout]() { return timeout.elapsed(); }, std::make_shared<SendViaSocket>(socket, hal));
}
private:
std::shared_ptr<ISocket> socket;
};
struct WaitConnectionEstablished : Func{
WaitConnectionEstablished(std::unique_ptr<IConnection> &conn, const HAL &hal) : Func(hal), conn(std::move(conn)){
}
std::shared_ptr<Func> run() override{
if (conn->PollSocketReady()){
auto ret = conn->PollSocketReady().value();
return std::make_shared<SendViaSocket>(ret, hal);
}
return std::make_shared<WaitConnectionEstablished>(conn, hal);
}
private:
std::unique_ptr<IConnection> conn;
};
struct OpenConnection : Func{
OpenConnection(const HAL &hal) : Func(hal) {}
std::shared_ptr<Func> run() override{
auto ret = hal.net.CreateConnectionRequest("example.com", 80, Protocol::TCP);
return std::make_shared<WaitConnectionEstablished>(ret, hal);
}
};
}
// namespace
void functionalTest(const HAL &hal){
testtime = Timeout(30s);
auto start1 = BlinkOn(hal);
auto start2 = OpenConnection(hal);
auto start3 = ReadSensor(hal);
auto start4 = CombineFunc(hal, start1.run(), start2.run());
auto start5 = CombineFunc(hal, start4.run(), start3.run());
auto f = start5.run();
while (!testtime.elapsed()){
f = f->run();
}
}
// ESched.cpp : Defines the entry point for the application.
//
#include <functional>
#include <vector>
#include "main.hpp"
#include "functional/functionalTest.hpp"
#include "threading/ThreadTest.hpp"
#include "timetriggered/TimeTriggered.hpp"
#include "statemachine/statemachineTest.hpp"
#include "coroutines/CoroutineTest.hpp"
#include "channels/ChannelsTest.hpp"
#include "publish_subscribe/PubSubTest.hpp"
#include "statically_scheduled/StaticallyScheduled.hpp"
using namespace std;
/** all tests have their entry point placed in here:
*/
struct Algo{
std::string name;
std::function<void(const HAL &hal)> func;
};
std::vector<Algo> algos = {
{"functional", functionalTest},
{"threads", threadTest},
{"timetriggered", timetriggeredTest},
{"statemachine", statemachineTest},
{"coroutines", coroutineTest},
{"channels", channelsTest},
{"pubsub", pubsubTest},
{"static", staticTest},
};
static void Usage(){
std::cerr << "Usage: ";
auto comma = "";
for (auto a : algos){
std::cerr << comma << a.name;
comma = ", ";
}
std::cerr << "\n";
}
int main(int argc, char **argv){
if (argc != 2){
std::cerr << "not enough arguments: " << std::to_string(argc) << "\n";
Usage();
return 1;
}
auto hal = CreateHAL();
for (auto a : algos){
if (a.name == argv[1]){
std::cerr << "Running algo: " << a.name << "\n";
a.func(hal);
return 0;
}
}
std::cerr << "Unknown algo: " << argv[1] << "\n";
Usage();
return 1;
}
#include "PubSubTest.hpp"
#include <thread>
#include <array>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include "PubSubInterfaces.hpp"
using namespace std::chrono_literals;
namespace{
Timeout testtime;
// -[ Handling Periodic Events ]-
template<int period>
class PeriodicPublisher : public Publisher{
public:
PeriodicPublisher(EventDispatcher& dispatcher) : Publisher(dispatcher) {}
class PeriodicEvent : public IEvent{
public:
class Type : public IEventType {
std::string ToString() const override{
return "periodic<" + std::to_string(period) + ">";
}
};
static std::shared_ptr<Type> periodicType;
std::shared_ptr<IEventType> GetType() override{
return periodicType;
}
static std::shared_ptr<IEventType> GetEventType(){
return periodicType;
}
};
};
using OneSecPublisher = PeriodicPublisher<1>;
using FiveSecPublisher = PeriodicPublisher<5>;
using OneSecPublisherEv = OneSecPublisher::PeriodicEvent;
using FiveSecPublisherEv = FiveSecPublisher::PeriodicEvent;
template<>
std::shared_ptr<OneSecPublisherEv::Type> OneSecPublisherEv::periodicType = std::make_shared<OneSecPublisherEv::Type>();
template<>
std::shared_ptr<FiveSecPublisherEv::Type> FiveSecPublisherEv::periodicType = std::make_shared<FiveSecPublisherEv::Type>();
// -[ Handling Sensor Values ]-
class SensorValueEvent : public IEvent{
private:
class Type : public IEventType {
std::string ToString() const override{
return "sensor event";
}
};
static std::shared_ptr<Type> type;
public:
std::shared_ptr<IEventType> GetType() override { return type; }
static std::shared_ptr<IEventType> GetEventType() { return type; }
SensorValueEvent(int8_t value) : value(value) {}
int8_t GetValue() { return value; }
private:
int8_t value;
};
std::shared_ptr<SensorValueEvent::Type> SensorValueEvent::type = std::make_shared<SensorValueEvent::Type>();
class SensorSubscriber : public ISubscriber, public Publisher{
public:
SensorSubscriber(ISensor<int8_t>& sensor, EventDispatcher& dispatcher)
: sensor(sensor), Publisher(dispatcher){
}
void HandleEvent(std::shared_ptr<IEvent> event){
auto value = sensor.GetValue();
Publish(std::make_shared<SensorValueEvent>(value));
}
private:
ISensor<int8_t>& sensor;
};
// -[ Handling the LED ]-
class LEDSubscriber : public ISubscriber{
public:
LEDSubscriber(ILED& led) : led(led){
}
void HandleEvent(std::shared_ptr<IEvent> event){
state = !state;
if (state)
led.on();
else
led.off();
}
private:
ILED& led;
bool state = false;
};
// -[ Handling the network ]-
class NetworkSubscriber : public ISubscriber{
public:
NetworkSubscriber(INetworkDevice& device) : device(device){
}
void HandleEvent(std::shared_ptr<IEvent> event){
if (!socket){
auto conn = device.CreateConnectionRequest("example.com", 80, Protocol::SSL);
while (!conn->PollSocketReady()){
}
socket = conn->PollSocketReady().value();
}
if (auto sensorEvent = std::dynamic_pointer_cast<SensorValueEvent>(event)){
payload += ", " + std::to_string(sensorEvent->GetValue());
}
else if (auto sensorEvent = std::dynamic_pointer_cast<FiveSecPublisherEv>(event)){
std::string msg = "payload: " + payload;
socket->Write(msg);
payload = "";
}
else{
std::cerr << "unknown event type received: " << event->GetType()->ToString() << std::endl;
}
}
private:
INetworkDevice& device;
std::shared_ptr<ISocket> socket;
std::string payload;
};
}
// namespace
void pubsubTest(const HAL& hal){
testtime = Timeout(30s);
EventDispatcher dispatcher;
OneSecPublisher oneSecPublisher(dispatcher);
FiveSecPublisher fiveSecPublisher(dispatcher);
// Sensor subscribes to 1 sec events and publishes sensor events
auto sensorSubs = std::make_shared<SensorSubscriber>(hal.sensor, dispatcher);
dispatcher.Subscribe(OneSecPublisher::PeriodicEvent::periodicType, sensorSubs);
// LED subscribes only to 1 sec events
auto ledSubs = std::make_shared<LEDSubscriber>(hal.led);
dispatcher.Subscribe(OneSecPublisher::PeriodicEvent::periodicType, ledSubs);
// network device subscribes to both sensor value and 5sec events
auto netSubs = std::make_shared<NetworkSubscriber>(hal.net);
dispatcher.Subscribe(FiveSecPublisher::PeriodicEvent::periodicType, netSubs);
dispatcher.Subscribe(SensorValueEvent::GetEventType(), netSubs);
// trigger the periodic events:
auto secPubThread = std::thread([&]() {
while (!testtime.elapsed()){
for (int i = 0; i < 4; i++){
oneSecPublisher.Publish(std::make_shared<OneSecPublisher::PeriodicEvent>());
std::this_thread::sleep_for(1s);
}
oneSecPublisher.Publish(std::make_shared<OneSecPublisher::PeriodicEvent>());
fiveSecPublisher.Publish(std::make_shared<FiveSecPublisher::PeriodicEvent>());
std::this_thread::sleep_for(1s);
}
});
secPubThread.join();
std::cerr << "exit!!!" << std::endl;
}
#include <queue>
#include <vector>
#include "statemachineTest.hpp"
#include "HAL.hpp"
using namespace std::chrono_literals;
class StateMachine{
public:
virtual void Step() = 0;
};
class OuterStateMachine : public StateMachine{
public:
void Step() override{
for (auto sm : machines){
sm->Step();
}
}
void Add(std::unique_ptr<StateMachine> sm){
machines.emplace_back(std::move(sm));
}
private:
std::vector<std::shared_ptr<StateMachine>> machines;
};
class LEDStateMachine : public StateMachine{
private:
enum class State{
INIT,
LED_ON,
LED_WAIT_ON,
LED_OFF,
LED_WAIT_OFF,
};
State state = State::INIT;
Timeout timeout;
ILED &led;
public:
LEDStateMachine(ILED &led) : led(led) {}
public:
void Step(){
switch (state){
case State::INIT:{
state = State::LED_ON;
break;
}
case State::LED_ON:{
state = State::LED_WAIT_ON;
timeout = Timeout(1s);
led.on();
break;
}
case State::LED_WAIT_ON:{
if (timeout.elapsed()){
state = State::LED_OFF;
}
break;
}
case State::LED_OFF:{
state = State::LED_WAIT_OFF;
timeout = Timeout(1s);
led.off();
break;
}
case State::LED_WAIT_OFF:{
if (timeout.elapsed()){
state = State::LED_ON;
}
break;
}
}
}
};
std::queue<int8_t> sensorValues;
class SensorStateMachine : public StateMachine{
private:
enum class State{
INIT,
ON,
WAIT_ON
};
State state = State::INIT;
Timeout timeout;
ISensor<int8_t> &sensor;
public:
SensorStateMachine(ISensor<int8_t> &sensor) : sensor(sensor) {}
public:
void Step(){
switch (state){
case State::INIT:{
state = State::ON;
break;
}
case State::ON:{
auto value = sensor.GetValue();
sensorValues.push(value);
state = State::WAIT_ON;
timeout = Timeout(1s);
break;
}
case State::WAIT_ON:{
if (timeout.elapsed()){
state = State::ON;
}
break;
}
}
}
};
class NetworkStateMachine : public StateMachine{
private:
enum class State{
INIT,
WAIT_CONN,
SEND,
WAIT_SEND,
};
State state = State::INIT;
Timeout timeout;
INetworkDevice &device;
std::unique_ptr<IConnection> conn;
std::shared_ptr<ISocket> socket;
public:
NetworkStateMachine(INetworkDevice &device) : device(device) {}
public:
void Step(){
switch (state){
case State::INIT:{
conn = std::move(device.CreateConnectionRequest("example.com", 80, Protocol::TCP));
state = State::WAIT_CONN;
break;
}
case State::WAIT_CONN:{
auto sock_opt = conn->PollSocketReady();
if (sock_opt){
socket = sock_opt.value();
state = State::WAIT_SEND;
timeout = Timeout(5s);
}
break;
}
case State::SEND:{
std::vector<int8_t> msg;
while (!sensorValues.empty()){
msg.push_back(sensorValues.front());
sensorValues.pop();
}
socket->Write(std::string("payload: ") + std::to_string(msg.size()) + ":" + std::string((char *)msg.data(), msg.size()));
state = State::WAIT_SEND;
timeout = Timeout(5s);
break;
}
case State::WAIT_SEND:{
if (timeout.elapsed()){
state = State::SEND;
}
break;
}
}
}
};
void statemachineTest(const HAL &hal){
auto testtime = Timeout(30s);
auto led_sm = std::make_unique<LEDStateMachine>(hal.led);
auto sensor_sm = std::make_unique<SensorStateMachine>(hal.sensor);
auto network_sm = std::make_unique<NetworkStateMachine>(hal.net);
OuterStateMachine sm;
sm.Add(std::move(led_sm));
sm.Add(std::move(sensor_sm));
sm.Add(std::move(network_sm));
while (!testtime.elapsed()){
sm.Step();
}
}
#include "StaticallyScheduled.hpp"
#include <thread>
#include <array>
#include <mutex>
#include <condition_variable>
using namespace std::chrono_literals;
namespace{
Timeout testtime;
}
void staticTest(const HAL& hal){
testtime = Timeout(30s);
}
// vtime 0
PAD_TIME_TAKEN( conn = create_connection_request, 1000)
// vtime 1000
PAD_TIME_TAKEN( wait is_ready(conn), 10000)
SKIP(990000)
// vtime 1001000
PAD_TIME_TAKEN( every 1000000 usec toggle_led, 1000)
// vtime 1002000
PAD_TIME_TAKEN( every 1000000 usec sample_sensor, 50000)
SKIP(950000)
// vtime 2002000
PAD_TIME_TAKEN( every 1000000 usec toggle_led, 1000)
// vtime 2003000
PAD_TIME_TAKEN( every 1000000 usec sample_sensor, 50000)
SKIP(950000)
// vtime 3003000
PAD_TIME_TAKEN( every 1000000 usec toggle_led, 1000)
// vtime 3004000
PAD_TIME_TAKEN( every 1000000 usec sample_sensor, 50000)
SKIP(950000)
// vtime 4004000
PAD_TIME_TAKEN( every 1000000 usec toggle_led, 1000)
// vtime 4005000
PAD_TIME_TAKEN( every 1000000 usec sample_sensor, 50000)
SKIP(946000)
// vtime 5001000
PAD_TIME_TAKEN( every 5000000 usec upload, 100000)
#include "ThreadTest.hpp"
#include <thread>
#include <queue>
#include <iostream>
#include <HAL.hpp>
#include <mutex>
#include "threadsafe_queue.hpp"
using namespace std::chrono_literals;
namespace{
Timeout testtime;
threadsafe_queue<int8_t> sensorValues;
void blinkThread(const HAL& hal){
while (true){
if (testtime.elapsed()) return;
hal.led.on();
std::this_thread::sleep_for(1s);
hal.led.off();
std::this_thread::sleep_for(1s);
}
}
void networkThread(const HAL& hal){
auto conn = hal.net.CreateConnectionRequest("example.com", 80, Protocol::TCP);
while (!conn->PollSocketReady().has_value()){
std::this_thread::sleep_for(100ms);
}
auto socket = conn->PollSocketReady().value();
while (true){
if (testtime.elapsed()) return;
std::vector<int8_t> msg;
while (auto elt = sensorValues.pop()){
msg.push_back(elt.value());
}
socket->Write(std::string("payload: ") + std::to_string(msg.size()) + ":" + std::string((char*)msg.data(), msg.size()));
std::this_thread::sleep_for(5s);
}
}
void sensorThread(const HAL& hal){
while (true){
if (testtime.elapsed()) return;
auto value = hal.sensor.GetValue();
sensorValues.push(value);
std::this_thread::sleep_for(1s);
}
}
}
void threadTest(const HAL& hal){
testtime = Timeout(30s);
std::thread t1([&hal]() { blinkThread(hal); });
std::thread t2([&hal]() { sensorThread(hal); });
std::thread t3([&hal]() { networkThread(hal); });
t1.join();
t2.join();
t3.join();
}
#include <thread>
#include <queue>
#include <iostream>
#include <HAL.hpp>
#include "TimeTriggered.hpp"
using namespace std::chrono_literals;
namespace{
class TimerTask{
public:
TimerTask(std::chrono::milliseconds _stepSize)
: stepSize(_stepSize){
}
bool IsRunnable() const{
return timeout.elapsed();
}
void Reschedule(){
timeout = Timeout(stepSize);
}
void SetStep(std::chrono::milliseconds stepSize){
this->stepSize = stepSize;
}
virtual void Run() = 0;
private:
std::chrono::milliseconds stepSize;
Timeout timeout;
};
class BlinkTask : public TimerTask{
public:
BlinkTask(std::chrono::milliseconds _stepSize, ILED& led)
: TimerTask(_stepSize), led(led){
}
virtual void Run(){
on = !on;
if (on){
led.on();
}
else{
led.off();
}
}
private:
bool on = false;
ILED& led;
};
std::queue<int8_t> sensorValues;
class SensorTask : public TimerTask{
public:
SensorTask(std::chrono::milliseconds _stepSize, ISensor<int8_t>& sensor)
: TimerTask(_stepSize), sensor(sensor){
}
virtual void Run(){
auto value = sensor.GetValue();
sensorValues.push(value);
}
private:
ISensor<int8_t>& sensor;
};
class NetworkTask : public TimerTask{
public:
NetworkTask(std::chrono::milliseconds _stepSize, INetworkDevice& networkDevice)
: TimerTask(_stepSize), sensor(sensor), networkDevice(networkDevice){
}
virtual void Run(){
if (! conn){
conn = std::move(networkDevice.CreateConnectionRequest("example.com", 80, Protocol::TCP));
}
if (!socket){
auto s = conn->PollSocketReady();
if (s){
socket = s.value();
SetStep(5s);
}
}
if (socket){
std::vector<int8_t> msg;
while (! sensorValues.empty()){
msg.push_back(sensorValues.front());
sensorValues.pop();
}
socket->Write(std::string("payload: ") + std::to_string(msg.size()) + ":" + std::string((char*)msg.data(), msg.size()));
}
}
private:
ISensor<int8_t>& sensor;
INetworkDevice& networkDevice;
std::unique_ptr<IConnection> conn;
std::shared_ptr<ISocket> socket;
};
std::vector<std::shared_ptr<TimerTask>> tasks;
void timerInterrupt(){
std::cerr << "tick\n";
for (auto& t : tasks){
if (t->IsRunnable()){
t->Reschedule();
t->Run();
}
}
}
}
void timetriggeredTest(const HAL& hal){
tasks.push_back(std::make_shared<BlinkTask>(1s, hal.led));
tasks.push_back(std::make_shared<SensorTask>(1s, hal.sensor));
tasks.push_back(std::make_shared<NetworkTask>(500ms, hal.net));
// at startup we use a 500msec freq. for connection setup
hal.SetTimerInterrupt(500ms, timerInterrupt);
auto test_timeout = Timeout(10s);
while (! test_timeout.elapsed()){
// We wait for interrupts here to allow an embedded processor to go to sleep mode for a little bit.
// This saves power. Wakeup from sleep mode is extremely quick for most embedded MCUs.
hal.WaitForInterupt();
}
hal.SetTimerInterrupt(0ms, nullptr);
}
