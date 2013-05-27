package trabajo.instruVision.realidadaumentada;

import java.nio.FloatBuffer;

import org.opencv.android.BaseLoaderCallback; 
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ActionBar.LayoutParams;
import android.graphics.PixelFormat;
import android.view.Menu;
import android.view.WindowManager;

public class MainActivity extends Activity implements CvCameraViewListener {
	
	
    // Variable para OpenGL
    private GLSurfaceView mGLSurfaceView;
	private Renderer renderer;

	// variable para OpenCV
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat                  mRgba;
	private Mat 				 projection;
	private Mat 				 modelview;
	FloatBuffer  fmodelview ;
	FloatBuffer  fprojection;
	
	@SuppressLint("NewApi")
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        // Cargamos el layout y a–adimos el listener de la c‡mara 
		setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.Visualizador_camara);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        
        // Creo el objeto para visualizar los elementos en 3D y los hago translucido
        // para poder pintar encima de la imagen
        this.mGLSurfaceView = new GLSurfaceView(this);
		addContentView(this.mGLSurfaceView, new LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT)); 
			
		mGLSurfaceView.setZOrderMediaOverlay(true);
		mGLSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
		this.renderer = new Renderer(this.getResources());
		this.mGLSurfaceView.setRenderer(this.renderer);
		mGLSurfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT);
        
	}
	
	// callback para la imagen de la c‡mara
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("realidad_aumentada");
                	mOpenCvCameraView.enableView();

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    //Creamos el menu, en este caso esta vacio
	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	//Cuando empieza el programa se instacia la variable que va a contener la imagen
	@Override
	public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(480, 640, CvType.CV_8UC3);
        projection = new Mat();
        modelview = new Mat();
        
		fmodelview = FloatBuffer.allocate(4*4);
		fprojection = FloatBuffer.allocate(4*4);
	}

	// pausa
    @Override
    public void onPause(){
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        super.onPause();
    }
	
    // cuando se cierre
    @Override
    public void onDestroy() {
        super.onDestroy(); 
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    // pararla
	@Override
	public void onCameraViewStopped() {
        mRgba.release();	
        modelview.release();
        projection.release();
       
	}
	
    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    // cuando se reciba un nueva frame
	@Override
	public Mat onCameraFrame(Mat inputFrame) {
		
		// se pasa como referencia el frame actual y las matrices de proyecci—n y modelview a la funci—n 
		// implementada en C con JNI
		augementedReallity(inputFrame.getNativeObjAddr(),
							projection.getNativeObjAddr(),
							modelview.getNativeObjAddr());
  	
		// se copia las matrices a un floatBuffer
	    for(int ix = 0; ix < modelview.cols(); ix++){
	        for(int iy = 0; iy < modelview.rows(); iy++){
	        	fmodelview.put(ix*modelview.rows() + iy,(float)modelview.get(iy, ix)[0]);
	        	fprojection.put(ix*projection.rows() + iy,(float)projection.get(iy, ix)[0]);
	        }
	    }
	    
	    // se para la matriz de proyeccion y de posicion y orientacion al visor de opengl
	    this.renderer.setMat(fmodelview, fprojection);
		
	    // se devuelve la imagen que va a ser representada en pantalla
		return inputFrame;
	}
    
	public native void augementedReallity(long matAddrRgba, long projection , long modelview);

}

