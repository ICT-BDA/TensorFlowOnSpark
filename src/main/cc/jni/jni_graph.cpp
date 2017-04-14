#include "include/bda_tensorflow_jni_Graph.h"
#include "util/util.h"


JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Graph_allocate
(JNIEnv * env, jobject jo, jobject jgd) {
	GraphDef* gd = (GraphDef*)GetNativeAddress(env, jgd);
	auto f_p = new FunctionLibraryDefinition(gd->library());
	Graph* g = new Graph(f_p);
	GraphConstructorOptions opts;
	Status status = ConvertGraphDefToGraph(opts, *gd, g);
	if (!status.ok()) {
		std::cout << "in c++ graph allocate method " << status.error_message() << std::endl;
	}
	SetNativeAddress(env, jo, g);
}

JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Graph_deallocateMemory
(JNIEnv * env, jobject jo, jlong jl) {
	delete (Graph*)jl;
}
