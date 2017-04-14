#include "include/bda_tensorflow_jni_Tensor.h"
#include "util/util.h"

/*
* Class:     tensorflow_jni_Tensor
* Method:    allocate
* Signature: (ILtensorflow/jni/TensorShape;)J
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Tensor_allocate
(JNIEnv * env, jobject obj, jint type, jobject jts) {
	TensorShape* ts = (TensorShape*)GetNativeAddress(env, jts);
	Tensor* t = new Tensor((DataType)type, *ts);
	SetNativeAddress(env, obj, t);
}

/*
* Class:     tensorflow_jni_Tensor
* Method:    deallocateMemory
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Tensor_deallocateMemory
(JNIEnv * env, jobject obj, jlong address) {
	delete (Tensor*)address;
}

/*
* Class:     tensorflow_jni_Tensor
* Method:    showContent
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Tensor_showContent
(JNIEnv * env, jobject jo) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	int size = t->NumElements();
	for (int i = 0; i < size; i++) {
		std::cout << t->flat<float>()(i) << std::endl;
	}
}

/*
* Class:     tensorflow_jni_Tensor
* Method:    toFloatArray
* Signature: ()[F
*/
JNIEXPORT jfloatArray JNICALL Java_bda_tensorflow_jni_Tensor_toFloatArray
(JNIEnv * env, jobject jo) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	int size = t->NumElements();
	jfloatArray jfa = env->NewFloatArray(size);
	env->SetFloatArrayRegion(jfa, 0, size, t->flat<float>().data());
	return jfa;
}

/*
* Class:     tensorflow_jni_Tensor
* Method:    initFromFloatArray
* Signature: ([F)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Tensor_initFromFloatArray
(JNIEnv * env, jobject jo, jfloatArray jfa) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	int size = t->NumElements();
	env->GetFloatArrayRegion(jfa, 0, size, t->flat<float>().data());
}

/*
* Class:     tensorflow_jni_Tensor
* Method:    getTensorShape
* Signature: ()[I
*/
JNIEXPORT jintArray JNICALL Java_bda_tensorflow_jni_Tensor_getTensorShape
(JNIEnv * env, jobject jo) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	const TensorShape& ts = t->shape();
	int dims = ts.dims();
	jintArray jia = env->NewIntArray(dims);
	for (int i = 0; i < dims; i++) {
		jint ji = ts.dim_size(i);
		env->SetIntArrayRegion(jia, i, 1, &ji);
	}
	return jia;

}

/*
* Class:     tensorflow_jni_Tensor
* Method:    toFloatArray
* Signature: ()[F
*/
JNIEXPORT jlongArray JNICALL Java_bda_tensorflow_jni_Tensor_toLongArray
(JNIEnv * env, jobject jo) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	int size = t->NumElements();
	jlongArray jla = env->NewLongArray(size);
	env->SetLongArrayRegion(jla, 0, size, (jlong*)t->flat<int64>().data());
	return jla;
}

/*
* Class:     tensorflow_jni_Tensor
* Method:    initFromFloatArray
* Signature: ([F)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Tensor_initFromLongArray
(JNIEnv * env, jobject jo, jlongArray jla) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	int size = t->NumElements();
	env->GetLongArrayRegion(jla, 0, size, (jlong*)t->flat<int64>().data());
}

JNIEXPORT jbooleanArray JNICALL Java_bda_tensorflow_jni_Tensor_toBooleanArray
(JNIEnv * env, jobject jo) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	//std::cout << t->flat<bool>()(0) << std::endl;
	int size = t->NumElements();
	jbooleanArray jba = env->NewBooleanArray(size);
	env->SetBooleanArrayRegion(jba, 0, size, (jboolean*)t->flat<bool>().data());
	return jba;
}

JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Tensor_initFromIntArray
(JNIEnv * env, jobject jo, jintArray jia) {
	Tensor* t = (Tensor*)GetNativeAddress(env, jo);
	int size = t->NumElements();
	env->GetIntArrayRegion(jia, 0, size, (jint*)t->flat<int>().data());
}
