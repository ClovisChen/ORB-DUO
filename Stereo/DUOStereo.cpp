#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<thread>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<DUOReader.h>

using namespace std;

void CALLBACK DUOCallback(const PDUOFrame pFrameData, void *pUserData) {
    PDUOFrame _pFrameData = pFrameData;
    DUOReader *duoReader = (DUOReader*)pUserData;
    unique_lock<mutex> lock(duoReader->mMutexCamera);
    duoReader->left.create(_pFrameData->height, _pFrameData->width, CV_8U);
    duoReader->right.create(_pFrameData->height, _pFrameData->width, CV_8U);
    duoReader->left.data = _pFrameData->leftData;
    duoReader->right.data = _pFrameData->rightData;
    duoReader->timeStamp = pFrameData->timeStamp;
    duoReader->ready = true;
}

void track_thread(void *reader, void* system){
    DUOReader *duo_reader = (DUOReader*)reader;
    ORB_SLAM2::System *slam_system = (ORB_SLAM2::System *)system;

    while (1)
    {
        if (duo_reader->ready == false)
            continue;
//        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        slam_system->TrackStereo(duo_reader->left, duo_reader->right, duo_reader->timeStamp);
//        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        cout<<ttrack<<endl;
        duo_reader->ready = false;
    }
}


#define WIDTH    752
#define HEIGHT    480
#define FPS        10

int main(int argc, char **argv) {

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    /// argv[1]: strVocFile  argv[2]: strSettingsFile
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, true);

    DUOReader duo_reader;
    duo_reader.OpenDUOCamera(WIDTH, HEIGHT, FPS);
    duo_reader.SetGain(0);
    duo_reader.SetExposure(100);
//    duo_reader.SetAutoExpose(true);
    duo_reader.SetLed(30);
    duo_reader.SetIMURate(200);
    duo_reader.SetUndistort(true);

    std::thread* mptTracking;
    mptTracking = new thread(track_thread, &duo_reader, &SLAM);

    duo_reader.StartDUOFrame(DUOCallback, &duo_reader);

//    SLAM.Shutdown();
//    duo_reader.CloseDUOCamera();
//    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}
