#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/imgproc.hpp>
#include <android/log.h>
#define  LOG_TAG    "your-log-tag"

#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

#define JNI_METHOD(NAME) \
    Java_com_example_eye_1detection_MainActivity_##NAME

//extern "C" JNIEXPORT jstring JNICALL
//Java_com_example_eye_1detection_MainActivity_stringFromJNI(
//        JNIEnv* env,
//        jobject /* this */) {
//}
void faceDetectionDlib(cv::Mat& img,cv::Mat& dst);

void rendertoMat(std::vector<dlib::full_object_detection>& dets, cv::Mat& dst);

extern "C"
JNIEXPORT void JNICALL
JNI_METHOD(LandmarkDetection)(JNIEnv* env,jobject,jlong addrInput,jlong addrOuput){

    cv::Mat& image = *(cv::Mat*)addrInput;
    cv::Mat& dst = *(cv::Mat*)addrOuput;
    //dst = image.clone();
    faceDetectionDlib(image,dst);

}

void faceDetectionDlib(cv::Mat& img,cv::Mat& dst) {
    try {
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("storage/emulated/0/shape_predictor_68_face_landmarks.dat") >> pose_model;;

        dlib::cv_image<dlib::bgr_pixel> cimg(img);
        //detect face
        std::vector<dlib::rectangle> faces = detector(cimg);
        //find pose
        std::vector<dlib::full_object_detection> shapes;

        for (unsigned long i = 0; i < faces.size(); ++i) {
            shapes.push_back(pose_model(cimg, faces[i]));
        }

        dst = img.clone();
        rendertoMat(shapes, dst);
    }catch(dlib::serialization_error& e) {
        //toDo
        LOGD("---------------- Error: %s",e.what());
        std::cout << std::endl << e.what() << std::endl;
    }
}

void rendertoMat(std::vector<dlib::full_object_detection>& dets, cv::Mat& dst) {
    cv::Scalar color;
    int sz=3;
    color = cv::Scalar(0,255,0);
    //chin line
    for(unsigned long idx=0;idx< dets.size();idx++){
        //left eye
        for(unsigned long i=37;i<41;++i){
            cv::line(dst,cv::Point(dets[idx].part(i).x(),dets[idx].part(i).y()),cv::Point(dets[idx].part(i-1).x(),dets[idx].part(i-1).y()),color,sz);
        }
        //right eye
        for(unsigned long i=43;i<47;++i){
            cv::line(dst,cv::Point(dets[idx].part(i).x(),dets[idx].part(i).y()),cv::Point(dets[idx].part(i-1).x(),dets[idx].part(i-1).y()),color,sz);
        }
    }

}
