#include "include/bda_tensorflow_util_Util.h"
#include "util/util.h"

/*
* Class:     tensorflow_util_cnn_AddAssignNodeToGraph
* Method:    addAssignNodeAccordingVariableName
* Signature: (Ltensorflow/jni/Graph;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_util_Util_addAssignNodeAccordingVariableName
(JNIEnv * env, 
	jclass cls,
	jobject jgraph, 
	jstring jvariableName,
	jstring jassignName, 
	jstring jplaceholderName, 
	jint jtype) {
	const char* variableName = env->GetStringUTFChars(jvariableName, 0);
	const char* assignName = env->GetStringUTFChars(jassignName, 0);
	const char* placeholderName = env->GetStringUTFChars(jplaceholderName, 0);

	Node* variableNode = NULL;
	Graph* graph = (Graph*)GetNativeAddress(env, jgraph);
	for (Node* node : graph->nodes()) {
		if (node->name() == variableName) {
			variableNode = node;
			break;
		}
	}
	// add placeholder node to graph
	NodeDef placeholderDef;
	placeholderDef.set_name(placeholderName);
	placeholderDef.set_op("Placeholder");  // N-way Add
	DataType dtype = (DataType)jtype;
	AddNodeAttr("dtype", dtype, &placeholderDef);
	Status s;
	Node* placeholderNode = graph->AddNode(placeholderDef, &s);
	TF_CHECK_OK(s);
	// add assign node to graph
	NodeDef assignDef;
	assignDef.set_name(assignName);
	assignDef.set_op("Assign");
	assignDef.add_input(variableName);
	assignDef.add_input(placeholderName);
	AddNodeAttr("T", dtype, &assignDef);
	Node* assignNode = graph->AddNode(assignDef, &s);
	TF_CHECK_OK(s);
	// add edges to the graph
	graph->AddEdge(variableNode, 0, assignNode, 0);
	graph->AddEdge(placeholderNode, 0, assignNode, 1);

	env->ReleaseStringUTFChars(jvariableName, variableName);
	env->ReleaseStringUTFChars(jassignName, assignName);
	env->ReleaseStringUTFChars(jplaceholderName, placeholderName);

}
