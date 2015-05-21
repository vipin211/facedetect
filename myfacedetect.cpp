#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <list>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

static void help() {
  cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
          "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
          "It's most known use is for faces.\n"
          "Usage:\n"
          "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
             "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
             "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
             "   [--try-flip]\n"
             "   [filename|camera_index]\n\n"
          "see facedetect.cmd for one call:\n"
          "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
          "During execution:\n\tHit any key to quit.\n"
          "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

// Detects faces in img and returns number of faces found.
size_t Detect(Mat& img, CascadeClassifier& cascade,
              CascadeClassifier& nestedCascade,
              double scale, bool tryflip);

// Processes directory for images underneath. Traverses directories that are
// inside.
void ProcessDirectory(const string& input_name, CascadeClassifier& cascade,
                      CascadeClassifier& nestedCascade,
                      double scale, bool tryflip);

string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main( int argc, const char** argv ) {
  CvCapture* capture = 0;
  Mat frame, frameCopy, image;
  const string scaleOpt = "--scale=";
  size_t scaleOptLen = scaleOpt.length();
  const string cascadeOpt = "--cascade=";
  size_t cascadeOptLen = cascadeOpt.length();
  const string nestedCascadeOpt = "--nested-cascade";
  size_t nestedCascadeOptLen = nestedCascadeOpt.length();
  const string tryFlipOpt = "--try-flip";
  size_t tryFlipOptLen = tryFlipOpt.length();
  string inputName;
  bool tryflip = false;

  help();

  CascadeClassifier cascade, nestedCascade;
  double scale = 1;

  for (int i = 1; i < argc; i++ ) {
    cout << "Processing " << i << " " <<  argv[i] << endl;
    if (cascadeOpt.compare(0, cascadeOptLen, argv[i], cascadeOptLen) == 0) {
      cascadeName.assign( argv[i] + cascadeOptLen );
      cout << "  from which we have cascadeName= " << cascadeName << endl;
    } else if(nestedCascadeOpt.compare(0, nestedCascadeOptLen, argv[i],
                                       nestedCascadeOptLen) == 0) {
      if (argv[i][nestedCascadeOpt.length()] == '=') {
          nestedCascadeName.assign(
              argv[i] + nestedCascadeOpt.length() + 1);
      }
      if (!nestedCascade.load( nestedCascadeName)) {
        cerr << "WARNING: Could not load classifier cascade for "
             << "nested objects" << endl;
      }
    } else if (scaleOpt.compare(0, scaleOptLen, argv[i], scaleOptLen) == 0) {
      if (!sscanf(argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1) {
        scale = 1;
      }
      cout << " from which we read scale = " << scale << endl;
    } else if (tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0) {
      tryflip = true;
      cout << " will try to flip image horizontally to detect assymetric objects\n";
    } else if( argv[i][0] == '-' ) {
      cerr << "WARNING: Unknown option %s" << argv[i] << endl;
    } else {
      inputName.assign( argv[i] );
    }
  }

  if (!cascade.load(cascadeName)) {
    cerr << "ERROR: Could not load classifier cascade" << endl;
    help();
    return -1;
  }

  if (inputName.empty() ||
      (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0')) {
    capture =
        cvCaptureFromCAM(inputName.empty() ? 0 : inputName.c_str()[0] - '0');
    int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;

    if (!capture) {
      cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
  }
  else if (inputName.size()) {
    image = imread( inputName, 1 );
    if (image.empty()) {
      capture = cvCaptureFromAVI( inputName.c_str() );
      if (!capture) {
        cout << "Capture from AVI didn't work" << endl;
      }
    }
  }
  else {
    image = imread( "../data/lena.jpg", 1 );
    if(image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
  }

  if (capture) {
    cout << "In capture ..." << endl;
    for (;;) {
      IplImage* iplImg = cvQueryFrame( capture );
      frame = cv::cvarrToMat(iplImg);
      if (frame.empty()) {
          break;
      }

      if (iplImg->origin == IPL_ORIGIN_TL) {
        frame.copyTo(frameCopy);
      } else {
        flip( frame, frameCopy, 0 );
      }

      Detect( frameCopy, cascade, nestedCascade, scale, tryflip );
      if (waitKey(10) >= 0) {
        goto _cleanup_;
      }
    }

    waitKey(0);
_cleanup_:
    cvReleaseCapture( &capture );
  } else {
    cout << "In image read" << endl;
    if (!image.empty()) {
      Detect( image, cascade, nestedCascade, scale, tryflip );
      waitKey(0);
    } else if (!inputName.empty()) {
      ProcessDirectory(inputName, cascade, nestedCascade, scale, tryflip);
    }
  }
  return 0;
}

void ProcessDirectory(const string& input_name, CascadeClassifier& cascade,
                      CascadeClassifier& nestedCascade,
                      double scale, bool tryflip) {
  DIR *dir = opendir(input_name.c_str());
  if (dir != NULL) {
    struct dirent *ent;
    std::list<std::string> dir_list;

    while ((ent = readdir(dir)) != NULL) {
      string file_name = string(ent->d_name);
      if (file_name == "." || file_name == "..")
        continue;
      string full_file_name = input_name + "/" + file_name;
      struct stat st;
      lstat(full_file_name.c_str(), &st);
      if (S_ISDIR(st.st_mode)) {
        dir_list.push_back(full_file_name);
        continue;
      }
      const string suf1 = ".jpg";
      const string suf2 = ".JPG";
      if (file_name.rfind(suf1) == (file_name.size() - suf1.size()) ||
          file_name.rfind(suf2) == (file_name.size() - suf2.size())) {
        size_t num_faces = 0;
        Mat image = imread(full_file_name.c_str(), 1 );
        if (!image.empty()) {
          num_faces = Detect(image, cascade, nestedCascade, scale, tryflip );
        } else {
            cerr << "Couldn't read image " << full_file_name << endl;
        }
        printf("%s, num faces: %lu\n", full_file_name.c_str(), num_faces);
      }
    }
    closedir (dir);

    for (list<string>::iterator it = dir_list.begin(); it != dir_list.end();
         ++it) {
      ProcessDirectory(*it, cascade, nestedCascade, scale, tryflip);
    }
  } else {
    perror ("Could not open directory");
  }
}

size_t Detect(Mat& img, CascadeClassifier& cascade,
              CascadeClassifier& nestedCascade,
              double scale, bool tryflip ) {
  double t = 0;
  vector<Rect> faces, faces2;
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY );

  Mat smallImg(cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1);
  resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
  equalizeHist(smallImg, smallImg );

  t = (double)cvGetTickCount();
  cascade.detectMultiScale(smallImg, faces,
                           1.1, 2, 0 |
                           // CASCADE_FIND_BIGGEST_OBJECT |
                           // CASCADE_DO_ROUGH_SEARCH |
                           CASCADE_SCALE_IMAGE,
                           Size(30, 30));
  if (tryflip) {
    flip(smallImg, smallImg, 1);
    cascade.detectMultiScale(smallImg, faces2,
                             1.1, 2, 0 |
                             // CASCADE_FIND_BIGGEST_OBJECT |
                             // CASCADE_DO_ROUGH_SEARCH |
                             CASCADE_SCALE_IMAGE,
                             Size(30, 30));
    for (vector<Rect>::const_iterator rr = faces2.begin(); rr != faces2.end();
         ++rr) {
      faces.push_back(Rect(smallImg.cols - rr->x - rr->width,
                           rr->y, rr->width, rr->height));
    }
  }
  t = (double)cvGetTickCount() - t;
  return faces.size();
}
