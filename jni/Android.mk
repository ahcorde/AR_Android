LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include /Users/ahcorde/programas/OpenCV-2.4.5-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := realidad_aumentada
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)
