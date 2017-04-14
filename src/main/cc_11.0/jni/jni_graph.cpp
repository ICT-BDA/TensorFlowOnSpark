#include "include/bda_tensorflow_jni_Graph.h"
#include "util/util.h"


JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Graph_allocate
(JNIEnv * env, jobject jo, jobject jgd) {
	GraphDef* gd = (GraphDef*)GetNativeAddress(env, jgd);
	auto f_p = new FunctionLibraryDefinition(OpRegistry::Global(), gd->library());
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

JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Graph_toGraphDef
(JNIEnv *env, jobject jgraph, jobject jgdf){
    GraphDef* gd = (GraphDef*)GetNativeAddress(env, jgdf);
    Graph* g = (Graph*)GetNativeAddress(env, jgraph);
    g->ToGraphDef(gd);
    SetNativeAddress(env, jgdf, gd);
}

