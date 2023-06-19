all: NormCECodelets.gp custom_ops.so merge_copies.so

NormCECodelets.gp: custom_ops/NormCECodelets.cpp
	popc --target=ipu2,ipu21 -DNDEBUG -O2 custom_ops/NormCECodelets.cpp custom_ops/NormCE.S custom_ops/NormCE_half_split.S -o NormCECodelets.gp

custom_ops.so: custom_ops/*.cpp
	g++ -std=c++14 -fPIC \
		-DONNX_NAMESPACE=onnx \
		custom_ops/TileMappingCommon.cpp \
		custom_ops/NormCE.cpp \
		custom_ops/NormCEImpl.cpp \
		custom_ops/softmax.cpp \
		custom_ops/kv_cache.cpp \
		custom_ops/beam_search_custom_op.cpp \
	        -shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so

merge_copies.so :
	g++ -std=c++14 -fPIC \
    -DONNX_NAMESPACE=onnx \
    custom_ops/merge_copies/custom_transform.cpp \
    -I custom_ops/merge_copies/ \
    -shared \
    -lpopart \
    -o merge_copies.so

.PHONY : clean
clean:
	-rm custom_ops.so NormCECodelets.gp merge_copies.so
