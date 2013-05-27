#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "/usr/local/include/eigen3/Eigen/Dense"


extern "C" {

	typedef struct {
			Eigen::Vector2d p2D;
			Eigen::Vector3d p3D;
		    int pattern_id;
	} TPatternPoint;

	class PatternDetector {
	public:
		/*
		 * Constructor de detector de patrones
		 * recibe como par‡metros del alto y ancho de la imagen
		 */
		PatternDetector(int IMAGE_HEIGHT, int IMAGE_WIDTH)
		{
			// valores de los umbrales
		    this->threshold1 = 13;
		    this->threshold2 = 3;

		    this->hsize = 56;
		    this->msize = 7;

		    // reserva de memoria de las im‡genes
		    this->filtered = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
		    this->camera_matrix = cv::Mat(3, 3, CV_32FC1);
		    this->dist_coeffs = cv::Mat(5, 1, CV_32FC1);
		    this->mapx = cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1);
		    this->mapy = cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1);

		    // matriz de los par‡metros intr’nsecos de la c‡mara
		    this->camera_matrix.at<float>(0,0) = 542.11200504548651;
		    this->camera_matrix.at<float>(0,1) = 0;
		    this->camera_matrix.at<float>(0,2) = 328.37445437402488;
		    this->camera_matrix.at<float>(1,0) = 0;
		    this->camera_matrix.at<float>(1,1) = 542.11;
		    this->camera_matrix.at<float>(1,2) = 247.040299;
		    this->camera_matrix.at<float>(2,0) = 0;
		    this->camera_matrix.at<float>(2,1) = 0;
		    this->camera_matrix.at<float>(2,2) = 1;

		    // parametros de distorsi—n
		    this->dist_coeffs.at<float>(0,0) = 0.0;
		    this->dist_coeffs.at<float>(1,0) = 0.0;
		    this->dist_coeffs.at<float>(2,0) = 0.0;
		    this->dist_coeffs.at<float>(3,0) = 0.0;
		    this->dist_coeffs.at<float>(4,0) = 0.0;

		    // inicializaci—n de los patrones
		    init_patterns();
		}

		/*
		 * Destructor. Liberaci—n de memoria necesaria para que la aplaci—n no explote
		 * tanto de las matrices como de los vectores que almacenan la informaci—n de los
		 * marcadores
		 * */

		~PatternDetector(){
			positions3D.clear();
			patterns.clear();

			dist_coeffs.release();
			camera_matrix.release();

		    detected.clear();
		    candidates.clear();

		    src_aux.release();
		    filtered.release();

		    mapx.release();
		    mapy.release();
		}


		/*
		 * Inicializa los patrones del tablero, con su codificaci—n y posici—n en metros
		 */
		void init_patterns()
		{
			//variables para almacenar el patron y la posici—n
		    Eigen::MatrixXd pattern;
		    Eigen::MatrixXd position;

		    // 1¼ Fila
		    // 1-1
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,1,1,0, 0,1,1,1,0, 1,0,1,1,1, 0,1,0,0,1, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0897,   -0.1365  ,       0,
		    		   -0.0507 ,  -0.1365  ,       0,
		    		   -0.0507 ,  -0.0975  ,       0,
		    		   -0.0897  , -0.0975  ,       0;
		    this->positions3D.push_back(position);

		    // 1-2
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,1,1,0, 1,0,1,1,1, 1,0,0,0,0, 1,0,1,1,1, 0,1,0,0,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0429 ,  -0.1365  ,       0,
		    		   -0.0039 ,  -0.1365  ,       0,
		    		   -0.0039 ,  -0.0975  ,       0,
		    		   -0.0429 ,  -0.0975  ,       0;
		    this->positions3D.push_back(position);

		    //1-3
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,1,1,0, 0,1,0,0,1, 1,0,0,0,0, 0,1,1,1,0, 1,0,0,0,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0039 ,  -0.1365  ,       0,
		    	    0.0429  , -0.1365    ,     0,
		    	    0.0429 ,  -0.0975    ,     0,
		    	    0.0039 ,  -0.0975    ,     0;
		    this->positions3D.push_back(position);

		    //1-4
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,1,1,1, 1,0,0,0,0, 0,1,0,0,1, 0,1,0,0,1, 0,1,1,1,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0507,   -0.1365,         0,
		    	    0.0897,   -0.1365     ,    0,
		    	    0.0897,   -0.0975      ,   0,
		    	    0.0507,   -0.0975       ,  0;
		    this->positions3D.push_back(position);

		    //2¼Fila
		    //2-1
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,1,1,1, 0,1,0,0,1, 0,1,0,0,1, 0,1,1,1,0, 1,0,0,0,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position <<    -0.0897 ,  -0.0897    ,     0,
		    		   -0.0507  , -0.0897    ,     0,
		    		   -0.0507 ,  -0.0507    ,     0,
		    		   -0.0897 ,  -0.0507    ,     0;
		    this->positions3D.push_back(position);

		    //2-2
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 0,1,0,0,1, 0,1,1,1,0, 1,0,0,0,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position <<    -0.0429 ,  -0.0897   ,      0,
		    		   -0.0039  , -0.0897   ,      0,
		    		   -0.0039 ,  -0.0507   ,      0,
		    		   -0.0429  , -0.0507   ,      0;
		    this->positions3D.push_back(position);

		    //2-3
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 1,0,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0039 ,  -0.0897  ,       0,
		    	    0.0429  , -0.0897     ,    0,
		    	    0.0429  , -0.0507    ,     0,
		    	    0.0039  , -0.0507    ,     0;
		    this->positions3D.push_back(position);

		    //2-4
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,1,1,1, 1,0,1,1,1, 1,0,1,1,1, 1,0,1,1,1, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0507 ,  -0.0897 ,        0,
		    	    0.0897 ,  -0.0897   ,      0,
		    	    0.0897 ,  -0.0507   ,      0,
		    	    0.0507 ,  -0.0507  ,       0;
		    this->positions3D.push_back(position);

		    //3¼ fila
		    //3-1
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,0,0,1, 0,1,1,1,0, 0,1,1,1,0, 0,1,0,0,1, 1,0,0,0,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position <<  -0.0897,   -0.0429  ,       0,
		    		   -0.0507 ,  -0.0429  ,       0,
		    		   -0.0507 ,  -0.0039  ,       0,
		    		   -0.0897 ,  -0.0039  ,       0;
		    this->positions3D.push_back(position);

		    //3-2
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern <<0,1,1,1,0, 1,0,1,1,1, 0,1,1,1,0, 1,0,0,0,0, 0,1,0,0,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0429  , -0.0429    ,     0,
		    		   -0.0039  , -0.0429   ,      0,
		    		   -0.0039 ,  -0.0039  ,       0,
		    		   -0.0429 ,  -0.0039   ,      0;
		    this->positions3D.push_back(position);

		    //3-3
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,1,1,0, 0,1,1,1,0, 1,0,1,1,1, 1,0,1,1,1, 0,1,0,0,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0039,   -0.0429,         0,
		    	    0.0429,   -0.0429  ,       0,
		    	    0.0429 ,  -0.0039  ,       0,
		    	    0.0039 ,  -0.0039 ,        0;
		    this->positions3D.push_back(position);

		    //3-4
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern <<0,1,1,1,0, 0,1,1,1,0, 1,0,1,1,1, 1,0,0,0,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0507  , -0.0429  ,       0,
		    	    0.0897  , -0.0429    ,     0,
		    	    0.0897  , -0.0039    ,     0,
		    	    0.0507  , -0.0039    ,     0;
		    this->positions3D.push_back(position);

		    //4¼fila
		    //4-1
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,1,1,1, 0,1,1,1,0, 1,0,1,1,1, 0,1,1,1,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position <<  -0.0897  ,  0.0039 ,        0,
		    		   -0.0507  ,  0.0039    ,     0,
		    		   -0.0507  ,  0.0429   ,      0,
		    		   -0.0897  ,  0.0429   ,      0;
		    this->positions3D.push_back(position);

		    //4-2
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 1,0,1,1,1, 0,1,1,1,0, 0,1,1,1,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position <<  -0.0429  ,  0.0039   ,      0,
		    		   -0.0039 ,   0.0039    ,     0,
		    		   -0.0039 ,   0.0429    ,     0,
		    		   -0.0429  ,  0.0429    ,     0;
		    this->positions3D.push_back(position);

		    //4-3
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,0,0,1, 0,1,1,1,0, 1,0,0,0,0, 0,1,1,1,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0039 ,   0.0039  ,       0,
		    	    0.0429  ,  0.0039     ,    0,
		    	    0.0429  ,  0.0429    ,     0,
		    	    0.0039  ,  0.0429      ,   0;
		    this->positions3D.push_back(position);

		    //4-4
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,1,1,0, 1,0,0,0,0, 1,0,1,1,1, 1,0,1,1,1, 0,1,1,1,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0507 ,   0.0039  ,       0,
		    	    0.0897 ,   0.0039   ,      0,
		    	    0.0897  ,  0.0429   ,      0,
		    	    0.0507  ,  0.0429   ,      0;
		    this->positions3D.push_back(position);

		    //5¼ fila
		    //5-1
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,0,0,1, 1,0,1,1,1, 0,1,0,0,1, 0,1,0,0,1, 0,1,0,0,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0897  ,  0.0507  ,       0,
		    		   -0.0507  ,  0.0507  ,       0,
		    		   -0.0507   , 0.0897 ,        0,
		    		   -0.0897    ,0.0897,         0;
		    this->positions3D.push_back(position);

		    //5-2
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 1,0,1,1,1, 1,0,0,0,0, 0,1,1,1,0, 1,0,0,0,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0429 ,   0.0507  ,       0,
		    		   -0.0039  ,  0.0507   ,      0,
		    		   -0.0039  ,  0.0897   ,      0,
		    		   -0.0429  ,  0.0897   ,      0;
		    this->positions3D.push_back(position);

		    //5-3
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 0,1,0,0,1, 0,1,1,1,0, 1,0,1,1,1, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0039   , 0.0507    ,     0,
		    	    0.0429  ,  0.0507      ,   0,
		    	    0.0429  ,  0.0897     ,    0,
		    	    0.0039  ,  0.0897      ,   0;
		    this->positions3D.push_back(position);

		    //5-4
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,1,1,0, 0,1,1,1,0, 0,1,0,0,1, 0,1,1,1,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0507  ,  0.0507,         0,
		    	    0.0897  ,  0.0507   ,      0,
		    	    0.0897   , 0.0897   ,      0,
		    	    0.0507  ,  0.0897   ,      0;
		    this->positions3D.push_back(position);

		    //6¼ fila
		    //6-1
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 0,1,0,0,1, 0,1,0,0,1, 0,1,1,1,0, 0,1,1,1,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0897   , 0.0975    ,     0,
		    		   -0.0507  ,  0.0975     ,    0,
		    		   -0.0507   , 0.1365     ,    0,
		    		   -0.0897   , 0.1365     ,    0;
		    this->positions3D.push_back(position);

		    //6-2
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 0,1,0,0,1, 0,1,0,0,1, 0,1,0,0,1, 0,1,1,1,0, 1,0,0,0,0;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << -0.0429 ,   0.0975    ,    0,
		    		   -0.0039  ,  0.0975     ,    0,
		    		   -0.0039  ,  0.1365    ,     0,
		    		   -0.0429  ,  0.1365     ,    0;
		    this->positions3D.push_back(position);

		    //6-3
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,0,0,0, 0,1,1,1,0, 0,1,0,0,1, 0,1,0,0,1, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0039 ,   0.0975,         0,
		    	    0.0429  ,  0.0975   ,      0,
		    	    0.0429  ,  0.1365   ,      0,
		    	    0.0039  ,  0.1365   ,      0;
		    this->positions3D.push_back(position);

		    //6-4
		    pattern = Eigen::MatrixXd(5, 5);
		    pattern << 1,0,1,1,1, 0,1,1,1,0, 1,0,0,0,0, 0,1,1,1,0, 1,0,1,1,1;
		    this->patterns.push_back(pattern);

		    position = Eigen::MatrixXd(4, 3);
		    position << 0.0507 ,   0.0975 ,        0,
		    	    0.0897 ,   0.0975    ,     0,
		    	    0.0897 ,   0.1365  ,       0,
		    	    0.0507 ,   0.1365   ,      0;
		    this->positions3D.push_back(position);

		}

		/*
		 * Umbralizaci—n de la imagen en el espacio HSV
		 */

		void filter_bw(cv::Mat &src) {
			for(int i=0; i<src.cols; i++) {
				for(int j=0; j<src.rows; j++) {
					cv::Vec3b pixel = src.at<cv::Vec3b>(j,i);
					if((int)(unsigned int)pixel[2] > (255*135.0/255.0))
						this->filtered.at<uchar>(j, i) = 1*255;
					else
						this->filtered.at<uchar>(j, i) = 0;
				}
			}
		}

		// distancia entre dos puntos en el plano
		double distanceBetweenPoints2D(int x1, int y1, int x2, int y2)
		{
			return sqrt( ((x2*x2)-(x1*x1)) + ((y2*y2)-(y1*y1)) );
		}


		// buscar rectangulos
		void search_rectangles(cv::Mat &src) {

			// contornos de la imagen
		    std::vector< std::vector<cv::Point> > contours;

		    // varaibles de apoyo
		    std::vector<cv::Vec4i> hierarchy;
		    std::vector<cv::Point> approximation;
		    int image_area = src.cols*2 + src.rows*2;
		    bool valid;
		    double dist, min_dist=image_area/200;
		    Eigen::MatrixXd mcandidate = Eigen::MatrixXd(10, 1);
		    double lx1, ly1, lx2, ly2;

		    src.copyTo(this->src_aux);

		    /*Encontrar los bordes o contornos de la imagen*/
		    cv::findContours(this->src_aux, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

		    /*comprobar los bordes detectaddos*/
		    for (int i=0;i<contours.size();i++ ) {

		      std::vector<cv::Point> contour =contours[i];
		      /*Se comprueba el tama–o para que no sean ni muy grande ni muy peque–os*/
		      if(contour.size() < image_area/20 || contour.size() > image_area/2)
		        continue;

		      /*Se comrpueba que sea un paralelogramo*/
		      cv::approxPolyDP(contour, approximation, double(contour.size())*0.1, true);
		      if(!(approximation.size()==4))
		        continue;

		      /*ÀEs convexo?*/
		      if(!cv::isContourConvex(cv::Mat(approximation)))
		        continue;

		      /*Distancia entre puntos no es demasiado grande*/
		      valid = true;
		      for(int j=0;j<4 && valid;j++) {
			      cv::Point p0 = approximation[j];
			      cv::Point p1 = approximation[(j+1)%4];

		        dist = distanceBetweenPoints2D(p0.x, p0.y, p1.x, p1.y);
		        if(dist < min_dist)
		          valid = false;
		      }
		      // si no valido se para al siguiente
		      if(!valid)
		        continue;

		      cv::Point p0 = approximation[0];
		      cv::Point p1= approximation[1];
		      cv::Point p2= approximation[2];
		      cv::Point p3= approximation[3];

		      /*Se comprubea la orientaci—n y se guardan en sentido antohoraio*/
		      lx1 = p1.x - p0.x;
		      ly1 = p1.y - p0.y;
		      lx2 = p2.x - p0.x;
		      ly2 = p2.y - p0.y;

		      if((lx1*ly2)-(ly1*lx2) >= 0.0)  {
		        mcandidate << p0.x, p0.y, p1.x, p1.y,
		                      p2.x, p2.y, p3.x, p3.y, -1, -1;
		      }else {
		        mcandidate << p0.x, p0.y, p3.x, p3.y,
		                      p2.x, p2.y, p1.x, p1.y, -1, -1;
		      }
		      /*Guardar*/
		      this->candidates.push_back(mcandidate);

		    }

		}

		// Comprbar que sean rectangulos

		void check_rectangles(cv::Mat &src) {
			// variables de apoyo
			Eigen::MatrixXd cmatrix;
			Eigen::MatrixXd hmatrix(8, 1);
			Eigen::MatrixXd bitmatrix(5, 5);
			cv::Mat hom, havg;
			cv::Point2f points1[4], points2[4];
			int msize = 7;
			int step= hsize/msize;
			int mcolor, color, index=0;
			bool valid;
			int val, orientation;

		    havg = cv::Mat(cv::Size(5,5), CV_8UC1);

		    hmatrix << 0, 0, hsize-1, 0, hsize-1, hsize-1, 0, hsize-1;
		    points2[0] = cv::Point2f(hmatrix(0,0),hmatrix(1,0));
		    points2[1] = cv::Point2f(hmatrix(2,0),hmatrix(3,0));
		    points2[2] = cv::Point2f(hmatrix(4,0),hmatrix(5,0));
		    points2[3] = cv::Point2f(hmatrix(6,0),hmatrix(7,0));

		    std::vector<Eigen::MatrixXd>::iterator it = this->candidates.begin();
		    while(it!=this->candidates.end()) {
		      cmatrix = (*it);

		      points1[0] = cv::Point2f(cmatrix(0,0),cmatrix(1,0));
		      points1[1] = cv::Point2f(cmatrix(2,0),cmatrix(3,0));
		      points1[2] = cv::Point2f(cmatrix(4,0),cmatrix(5,0));
		      points1[3] = cv::Point2f(cmatrix(6,0),cmatrix(7,0));

		      /*Crear la homografia*/
		      cv::Mat pers = cv::getPerspectiveTransform(points1, points2);
		      cv::warpPerspective(src, hom, pers, cv::Size(hsize, hsize), cv::INTER_NEAREST);

		      /*Los bordes tienen que ser del mismo color*/
		      valid = true;
		      for(int row=0;row<this->msize && valid;row++) {
		        for(int col=0;col<this->msize && valid;col++) {
		          if(row!=0 && row!=msize-1 && col!=0 && col!=msize-1)
		            continue;

		          /*Se guarda el primer colo*/
		          if(row==0 && col==0)
		            mcolor = this->get_color(hom, row, col, step);
		          else {
		            /*Se compara con ese primer color*/
		            color = this->get_color(hom, row, col, step);
		            valid = (color == mcolor);
		          }
		        }
		      }

		      /*No es valido. Se elimina*/
		      if(!valid) {
		        it = this->candidates.erase(it);
		        continue;
		      }

		      /*Creacion de la medio desde la homografia*/
		      for(int row=1;row<this->msize-1 && valid;row++) {
		    	  for(int col=1;col<this->msize-1 && valid;col++) {
		    		  color = this->get_color(hom, row, col, step);
		    		  if(color != mcolor) {
		    			  havg.at<uchar>(col-1,row-1) = 1*255;
		    			  bitmatrix(col-1,row-1) = 1;
		    		  } else {
		    			  havg.at<uchar>(col-1,row-1) = 0;
		    			  bitmatrix(col-1,row-1) = 0;
		    		  }
		    	  }
		      }

		      /*Se compara con los patrones guardados para esta orientacion */
		      val = -1;
		      for(int rot=0;rot<4 && val<0;rot++) {
		    	  val = this->search_pattern(bitmatrix);
		    	  orientation = rot;
		    	  this->turn_pattern_right(bitmatrix);
		      }

		      if(val < 0) {
		    	  it = this->candidates.erase(it);
		    	  continue;
		      }

		      /*Se guarda el patron detectado*/
		      (*it)(8,0) = (double) val;
		      (*it)(9,0) = (double) orientation;

		      it++;
		    }

		  }

		/*Giara el patron hacia la derecha -> para comprobar la orientaci—n 0, 90, 180 o 270 */

		void turn_pattern_right(Eigen::MatrixXd &pin) {
		    Eigen::MatrixXd mcol(5, 1);
		    int i2;
		    pin.transposeInPlace();

		    /*Swap colums*/
		    for(int i=0;i<2;i++) {
		      if(i==0) i2=4;
		      if(i==1) i2=3;

		      double d1 = pin(0, i);
		      double d2 = pin(1, i);
		      double d3 = pin(2, i);
		      double d4 = pin(3, i);
		      double d5 = pin(4, i);
		      mcol <<d1, d2, d3, d4, d5;

		      pin(0, i) = pin(0, i2);
		      pin(1, i) = pin(1, i2);
		      pin(2, i) = pin(2, i2);
		      pin(3, i) = pin(3, i2);
		      pin(4, i) = pin(4, i2);

		      pin(0, i2) = mcol(0, 0);
		      pin(1, i2) = mcol(1, 0);
		      pin(2, i2) = mcol(2, 0);
		      pin(3, i2) = mcol(3, 0);
		      pin(4, i2) = mcol(4, 0);
		    }
		}

		// busqueda del patron
		int search_pattern(Eigen::MatrixXd p) {
		    int size = this->patterns.size();
		    bool cancel[size];
		    bool found = true;

		    //memset(cancel, 0, size*sizeof(bool));
		    for(int i = 0; i < size;i++){
		    	cancel[i] = 0;
		    }
		    for(int i=0;i<5 && found;i++) {
		    	for(int j=0;j<5 && found;j++) {
		    		found = false;
		    		/*Se compara con los patrones*/
		    		int k=0;
		    		for(std::vector<Eigen::MatrixXd>::iterator it = this->patterns.begin(); it!= this->patterns.end(); it++) {
		    			if(cancel[k]==0) {
		    				if(p(i,j) == (*it)(i,j))
		    					found = true;
		    				else
		    					cancel[k] = true;
		    			}
		    			k++;
		    		}
		    	}
		    }

		    /*Patron encontrado*/
		    if(found) {
		    	for(int i=0;i<size;i++) {
		    		if(cancel[i]==0)
		    			return i;
		      }
		    }

		    return -1;
		  }

		// color
		int get_color(cv::Mat &src, int row, int col, int size) {
		    int rs, cs, tot, max;

		    rs = row*size;
		    cs = col*size;

		    /*Count how many rectangles are zero*/
		    cv::Mat box = src(cv::Rect(rs,cs,size,size));
		    tot = cv::countNonZero(box);

		    /*Calc average*/
		    max = (size*size)/2;
		    if (tot > max)
		    	return 1;
		    else
		    	return 0;
		  }

		bool detect(cv::Mat &src2)
		{
			// auxiliares
		    cv::Mat srcdist = cv::Mat(1, 1, CV_32FC2);
		    cv::Mat dstdist = cv::Mat(1, 1, CV_32FC2);
		    int index, orientation;
		    double rx, ry;

		    this->detected.clear();
		    this->candidates.clear();

		    // se umbraliza la imagen
		    this->filter_bw(src2);

		    // busqueda de rectangulos
		    this->search_rectangles(this->filtered);

		    // se comprueba que sean rectangulos
		    this->check_rectangles(this->filtered);

		    // para uno de los candidatos
		    for(int i=0;i<this->candidates.size();i++) {
		        index = (int) this->candidates[i](8,0);
		        orientation = (int) this->candidates[i](9,0);
		        for(int j=0;j<4;j++) {
		          TPatternPoint pp;

		          /* Si se dispone de los coeficientes de distorsi—n se puede suprimir la distorsi—n */
		          //srcdist.at<cv::Vec2f>(0,0)[0] = (double) this->candidates[i](2*j,0);
		          //srcdist.at<cv::Vec2f>(0,0)[1] = (double) this->candidates[i](2*j+1,0);
		          //cv::undistortPoints(srcdist, dstdist, this->camera_matrix, this->dist_coeffs);
		          //rx = dstdist.at<cv::Vec2f>(0,0)[0]*721.3920000000001+322.383;
		          //ry = dstdist.at<cv::Vec2f>(0,0)[1]* 721.208+179.418;
		          //pp.p2D << rx, ry;

		          pp.p2D << (double) this->candidates[i](2*j,0),  (double) this->candidates[i](2*j+1,0);
		          pp.p3D << this->positions3D[index]((j+orientation)%4, 0), this->positions3D[index]((j+orientation)%4, 1), this->positions3D[index]((j+orientation)%4, 2);
		          this->detected.push_back(pp);
		        }
		    }
		    return true;
		}

		// devulelve la lista con los patrones encontrados
	    std::vector<TPatternPoint> * getDetected() {
	        return &(this->detected);
	    }

	    // elimina la distorsi—n de la imagen
	    cv::Mat rectify_image(cv::Mat src) {
	      cv::initUndistortRectifyMap(this->camera_matrix, this->dist_coeffs, cv::Mat(), this->camera_matrix, src.size(), CV_32FC1, this->mapx, this->mapy);
	      cv::remap(src, src, this->mapx, this->mapy, cv::INTER_LINEAR);
	      return src;
	    }

	public:
		//Variables
	    std::vector<Eigen::MatrixXd> patterns;
	    std::vector<Eigen::MatrixXd> positions3D;
	    std::vector<TPatternPoint> detected;
	    std::vector<Eigen::MatrixXd> candidates;

	    cv::Mat src_aux;
	    cv::Mat filtered;

	    double threshold1;
	    double threshold2;

	    int hsize;  /*Tama–o de la homografia*/
	    int msize;  /*Tama–o del patron*/

	    /*Distortions*/
	    cv::Mat mapx;
	    cv::Mat mapy;
	    cv::Mat camera_matrix;
	    cv::Mat dist_coeffs;

	};

	/*
	 * Genera la matriz de proyecci—n y modelview
	 * La matriz de projecci—n de genera utilizando los par‡metros intrinsecos de la c‡mara
	 * EL modelview conocmiendo la posici—n y orientaci—n de la c‡mara
	 */
	void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview)
	{

	    const float zNear = 0.01;			// Distance to the OpenGL near clipping plane.
	    const float zFar = 1000.0;			// Distance to the OpenGL far clipping plane.

	    // matriz de proejccion
	    projection.at<double>(0,0) = 2*calibration.at<float>(0,0)/1024;
	    projection.at<double>(1,0) = 0;
	    projection.at<double>(2,0) = 0;
	    projection.at<double>(3,0) = 0;

	    projection.at<double>(0,1) = 0;
	    projection.at<double>(1,1) = 2*calibration.at<float>(1,1)/552;
	    projection.at<double>(2,1) = 0;
	    projection.at<double>(3,1) = 0;

	    projection.at<double>(0,2) = 1-2*calibration.at<float>(0,2)/1024;
	    projection.at<double>(1,2) = -1+(2*calibration.at<float>(1,2)+2)/552;
	    projection.at<double>(2,2) = (zNear+zFar)/(zNear - zFar);
	    projection.at<double>(3,2) = -1;

	    projection.at<double>(0,3) = 0;
	    projection.at<double>(1,3) = 0;
	    projection.at<double>(2,3) = 2*zNear*zFar/(zNear - zFar);
	    projection.at<double>(3,3) = 0;

	    // modelview
	    modelview.at<double>(0,0) = rotation.at<double>(0,0);
	    modelview.at<double>(1,0) = rotation.at<double>(1,0);
	    modelview.at<double>(2,0) = rotation.at<double>(2,0);
	    modelview.at<double>(3,0) = 0;

	    modelview.at<double>(0,1) = rotation.at<double>(0,1);
	    modelview.at<double>(1,1) = rotation.at<double>(1,1);
	    modelview.at<double>(2,1) = rotation.at<double>(2,1);
	    modelview.at<double>(3,1) = 0;

	    modelview.at<double>(0,2) = rotation.at<double>(0,2);
	    modelview.at<double>(1,2) = rotation.at<double>(1,2);
	    modelview.at<double>(2,2) = rotation.at<double>(2,2);
	    modelview.at<double>(3,2) = 0;

	    modelview.at<double>(0,3) = translation.at<double>(0,0);
	    modelview.at<double>(1,3) = translation.at<double>(1,0);
	    modelview.at<double>(2,3) = translation.at<double>(2,0);
	    modelview.at<double>(3,3) = 1;

	    // cambio de coordenadas por openGL
	    static double changeCoordArray[4][4] = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
	    static cv::Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

	    modelview = changeCoord*modelview;
	}

	JNIEXPORT void JNICALL Java_trabajo_instruVision_realidadaumentada_MainActivity_augementedReallity(JNIEnv*, jobject, jlong addrRgba, jlong addrProjection, jlong addrModelview);

	JNIEXPORT void JNICALL Java_trabajo_instruVision_realidadaumentada_MainActivity_augementedReallity(JNIEnv*, jobject, jlong addrRgba, jlong addrProjection, jlong addrModelview)
	{

		// se obtienen la matrices desde Java a C
		cv::Mat& mRgb = *(cv::Mat*)addrRgba;
		cv::Mat& projection = *(cv::Mat*)addrProjection;
		cv::Mat& modelview = *(cv::Mat*)addrModelview;
	    cv::Mat hsv;

	    // reserva de memoria de las matrices
	    modelview.create(4, 4, CV_64FC1);
	    projection.create(4, 4, CV_64FC1);

	    // Imagen a HSV
	    cv::cvtColor(mRgb, hsv, CV_RGB2HSV);

	    // instancia de la clase
	    PatternDetector* pdetector = new PatternDetector(mRgb.rows, mRgb.cols);

	    // SI se encuentra nada se librea memoraia y se sale de la funci—n
	    if(!pdetector->detect(hsv)){
		    hsv.release();
		    delete pdetector;
	        return;
	    }

	    // en casa de conocer los par‡metros de distorsi—n se elimina de la imagen
	    //mRgb = pdetector->rectify_image(mRgb);

	    // lista con los patrones detectados
	    std::vector<TPatternPoint>* patterns = pdetector->getDetected();

	    // puntos 2D de la imagen
	    std::vector<cv::Point2f> realCorners;
	    // puntos 3D a los que corresponde los puntos anteriores
	    std::vector<cv::Point3f> virtualCorners;		// The corresponding corner positions for where the corners lie on the chess board (measured in virtual units).

	    // se dibujan los puntos 2D detectados
	    int k = 0;
	    for(std::vector<TPatternPoint>::iterator it = patterns->begin(); it!= patterns->end();it++) {
			k++;
			realCorners.push_back(cv::Point2f((*it).p2D(0), (*it).p2D(1)));
			virtualCorners.push_back(cv::Point3f((*it).p3D(0) , (*it).p3D(1) , (*it).p3D(2) ));
			if(k%4==0)
				cv::circle(mRgb, cv::Point( (*it).p2D(0),   (*it).p2D(1) ), 2, cv::Scalar(255, 0, 0), 2);
			if(k%4==1)
				cv::circle(mRgb, cv::Point( (*it).p2D(0),   (*it).p2D(1) ), 2, cv::Scalar(0, 255, 0), 2);
			if(k%4==2)
				cv::circle(mRgb, cv::Point( (*it).p2D(0),   (*it).p2D(1) ), 2, cv::Scalar(0, 0, 255), 2);
			if(k%4==3)
				cv::circle(mRgb, cv::Point( (*it).p2D(0),   (*it).p2D(1) ), 2, cv::Scalar(255, 255, 0), 2);
		}

	    // liberaci—n de mekoria de imagen principal
	    hsv.release();

	    // si no hay patrones se sale
	    if(!realCorners.size()>0){
		    delete pdetector;
	    	return;
	    }

	    // matrices para conocere la orientaci—n y posici—n de la c‡mara
	    cv::Mat rotation;
	    cv::Mat translation;

	    // se resuelve la posici—n y orientaci—n de la camara conocientos los puntos 2D y a que puntos
	    // 3D pertenecen. Es necesario conocer la calibraci—n de la c‡mara
        cv::solvePnPRansac(cv::Mat(virtualCorners), cv::Mat(realCorners),
        		pdetector->camera_matrix, pdetector->dist_coeffs,
        		rotation, translation);

        // se pasa a matriz de rotaci—n
        cv::Mat rotationMatrix;
        cv::Rodrigues(rotation, rotationMatrix);

        // se puede a–adir un offset a la translaci—n en este caso es cero
        double offsetA[3][1] = { {0}, {0.0}, {0}};
        cv::Mat offset(3, 1, CV_64FC1, offsetA);
	    translation = translation + rotationMatrix*offset;

	    // se fgenera la matriz de porjeccion y modelview
	    generateProjectionModelview(pdetector->camera_matrix, rotationMatrix, translation, projection, modelview);

	    // Liberaci—n de la memoria
	    rotation.release();
	    translation.release();
	    virtualCorners.clear();
	    realCorners.clear();
	    delete pdetector;
	}
}
