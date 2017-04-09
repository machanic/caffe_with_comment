/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
//see http://alanse7en.github.io/caffedai-ma-jie-xi-4
/**
这个类的构造函数是private的，也就是用我们没有办法去构造一个这个类型的变量，
这个类也没有数据成员，所有的成员函数也都是static的，可以直接调用

我们首先从CreateSolver函数(第15行)入手，这个函数先定义了string类型的变量type，表示Solver的类型(‘SGD’/’Nestrov’等)，然后定义了一个key类型为string，value类型为Creator的map：registry，其中Creator是一个函数指针类型，指向的函数的参数为SolverParameter类型，返回类型为Solver<Dtype>*(见第2行和第3行)。如果是一个已经register过的Solver类型，那么registry.count(type)应该为1，然后通过registry这个map返回了我们需要类型的Solver的creator，并调用这个creator函数，将creator返回的Solver<Dtype>*返回。

上面的代码中，Registry这个函数（第5行）中定义了一个static的变量g_registry，这个变量是一个指向CreatorRegistry这个map类型的指针，然后直接返回，因为这个变量是static的，所以即使多次调用这个函数，也只会定义一个g_registry，而且在其他地方修改这个map里的内容，是存储在这个map中的。事实上各个Solver的register的过程正是往g_registry指向的那个map里添加以Solver的type为key，对应的Creator函数指针为value的内容。
**/
template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();//注意是static变量。其实重复执行，也只有一个g_register_
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();  
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
    const string& type = param.type();
    CreatorRegistry& registry = Registry();  //建立一个map<string, Creator>，其中Creator为第56行定义的函数指针
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    return registry[type](param);
  }

  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry() {}

  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};


template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};


/***
在sgd_solver.cpp(SGD Solver对应的cpp文件)末尾有上面第24行的代码，
使用了REGISTER_SOLVER_CLASS这个宏，这个宏会定义一个名为Creator_SGDSolver的函数，这个函数即为Creator类型的指针指向的函数，
在这个函数中调用了SGDSolver的构造函数，并将构造的这个变量得到的指针返回，
这也就是Creator类型函数的作用：构造一个对应类型的Solver对象，将其指针返回。
然后在这个宏里又调用了REGISTER_SOLVER_CREATOR这个宏，
这里分别定义了SolverRegisterer这个模板类的float和double类型的static变量，这会去调用各自的构造函数，
而在SolverRegisterer的构造函数中调用了之前提到的SolverRegistry类的AddCreator函数，这个函数就是将刚才定义的Creator_SGDSolver这个函数的指针存到g_registry指向的map里面。
类似地，所有的Solver对应的cpp文件的末尾都调用了这个宏来完成注册，在所有的Solver都注册之后，我们就可以通过之前描述的方式，通过g_registry得到对应的Creator函数的指针，并通过调用这个Creator函数来构造对应的Solver。
***/
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe，末尾调用宏REGISTER_SOLVER_CREATOR，传入刚刚定义的函数指针

#endif  // CAFFE_SOLVER_FACTORY_H_
