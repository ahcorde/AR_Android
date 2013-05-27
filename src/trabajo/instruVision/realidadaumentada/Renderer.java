package trabajo.instruVision.realidadaumentada;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import trabajo.instruVision.opengl.ObjReader;

import android.content.res.Resources;

import android.opengl.GLSurfaceView;
import android.opengl.GLU;

public class Renderer implements GLSurfaceView.Renderer{


	// los ejes X, Y, Z, tienen que salir en el centro de tablero
	float EjeT[] = new float[]{
			0f, 0f, 0.0f, 
			0.1f, 0f, 0.0f,
			0f, 0f, 0.0f,
			0f, 0.1f, 0.0f, 
			0f, 0f, 0.0f,
			0f, 0f, 0.1f,
	};
	FloatBuffer ejeTBuff;

	// variables de los modelos
	//superman
    private ObjReader.Model superman = null;
    boolean superman_read = false;
    
    // coche-> problemas muy grande y peta por memoria
    private ObjReader.Model molecula_glusosa = null;
    boolean molecula_glusosa_read = false;
    
    // Gundam, personaje de transformers
    private ObjReader.Model robot = null;
    boolean robot_read = false;
    
    // el batmovil
    private ObjReader.Model batmovil = null;
    boolean batmovil_read = false;
	
    
    // matrices para representar los objetos en 3D
	Resources resources;
	FloatBuffer modelview;
	FloatBuffer projectioMat;
	
	// anchura y altura de pantalla
	int width;
	int height;
	
	public Renderer(Resources resources){
		this.resources = resources;
				
		// transformar los ejes en un floatBuffer
		ejeTBuff = makeFloatBuffer(EjeT);
		
		// reservo memoria para la matrices de projecion y modelview
		this.modelview = FloatBuffer.allocate(4*4);
		this.projectioMat = FloatBuffer.allocate(4*4);

 
		// hilos para cargar los modelos 3D en formato obj
        Thread supermanThread= new Thread(){
			public void run(){
		        String ObjFileName_world = new String("/storage/sdcard0/Models/SupermanSmall.obj");

		        superman = ObjReader.ReadObj(ObjFileName_world );
		        if(superman!=null)	
		        	superman_read = true;
			}
		};
		supermanThread.start();
		
        Thread robotThread= new Thread(){
			public void run(){
		        String ObjFileName_world = new String("/storage/sdcard0/Models/Gundamobj2.obj");
		        robot = ObjReader.ReadObj(ObjFileName_world );
		        if(robot!=null)
		        	robot_read = true;
			}
		};
		robotThread.start();
		
        Thread batmovilThread= new Thread(){
			public void run(){
		        String ObjFileName = new String("/storage/sdcard0/Models/batmobile.obj");

		        batmovil = ObjReader.ReadObj(ObjFileName);

		        if(batmovil!=null)
		        	batmovil_read = true;
		       
			}
		};
		batmovilThread.start();
		
        Thread cocheThread= new Thread(){
			public void run(){
		        String ObjFileName = new String("/storage/sdcard0/Models/glucosa.obj");

		        molecula_glusosa = ObjReader.ReadObj(ObjFileName);

		        if(molecula_glusosa!=null)
		        	molecula_glusosa_read = true;
		        System.out.println("FINNNNN");
		       
			}
		};
		cocheThread.start();	
		
	}
	
	// metodo para copiar la matrices del algormito de visi—n y ser utilizadas por opengl
	public void setMat(FloatBuffer modelview, FloatBuffer projectioMat ){
	    for(int ix = 0; ix < 4; ix++){
	        for(int iy = 0; iy < 4; iy++){
	        	this.modelview.put(ix*4 + iy,(float)modelview.get(ix*4 + iy));
	        	this.projectioMat.put(ix*4 + iy,(float)projectioMat.get(ix*4 + iy));
	        }
	    }
	}
	
	// funci—n que copia un array de float a un floatbuffer
    FloatBuffer makeFloatBuffer(float[] arr) {
        ByteBuffer bb = ByteBuffer.allocateDirect(arr.length*4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(arr);
        fb.position(0);
        return fb;
    }
    
	// cada vez que tenga que dibujar en pantalla un nuevo frame
	@Override
	public void onDrawFrame(GL10 gl) {
	
		// inicializaci—n de opengl
		gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);

		// carga de las matrices de projection y modelviw
		gl.glMatrixMode(GL10.GL_PROJECTION);
		gl.glLoadIdentity();
		GLU.gluOrtho2D(gl, 0.0f, 1.0f, 0.0f, 1.0f);
					
		gl.glMatrixMode(GL10.GL_PROJECTION);
		gl.glLoadMatrixf(this.projectioMat);
		
		gl.glMatrixMode(GL10.GL_MODELVIEW);
		gl.glLoadMatrixf(this.modelview);
		
		// rotar y transladar para verlo correctamente en la pantalla
        gl.glRotatef(180.f, -1.0f, 0.0f, 0.0f);
        gl.glRotatef(-90.f, 0.0f, 0.0f, 1.0f);
        
		// Dibujar los ejes
		gl.glVertexPointer(3, GL10.GL_FLOAT, 0, ejeTBuff);
		gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
		gl.glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
		gl.glDrawArrays(GL10.GL_LINES, 0, 2);
		
		gl.glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
		gl.glDrawArrays(GL10.GL_LINES, 2, 2);
		
		gl.glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
		gl.glDrawArrays(GL10.GL_LINES, 4, 2);
		
		
		//Representar los modelos 3D
		
		if(superman_read){
			gl.glPushMatrix();
			gl.glTranslatef(0.1f, 0, 0);
			gl.glRotatef(90, 1, 0, 0);
			gl.glScalef(0.001f, 0.001f, 0.001f);
			superman.Draw(gl);
			gl.glPopMatrix();
		}
		
		if(robot_read){
			gl.glPushMatrix();
			gl.glTranslatef(-0.1f, 0, 0);
			gl.glRotatef(90, 1, 0, 0);
			gl.glScalef(0.01f, 0.01f, 0.01f);
			robot.Draw(gl);
			gl.glPopMatrix();
		}
		
		if(batmovil_read){
			gl.glPushMatrix();
			gl.glTranslatef(0.0f, 0.2f, 0);
			gl.glRotatef(90, 1, 0, 0);
			gl.glScalef(0.01f, 0.01f, 0.01f);
			batmovil.Draw(gl);
			gl.glPopMatrix();
		}
		
		
		if(molecula_glusosa_read){
			gl.glPushMatrix();
			gl.glTranslatef(0.0f, -0.2f, 0);
			gl.glRotatef(90, 1, 0, 0);
			gl.glScalef(0.03f, 0.03f, 0.03f);
			molecula_glusosa.Draw(gl);
			gl.glPopMatrix();
		}
	}

	// si cambia 
	@Override
	public void onSurfaceChanged(GL10 gl, int w, int h) {		
    	this.width = w;
    	this.height = h;
		
		gl.glMatrixMode(GL10.GL_PROJECTION);
		gl.glLoadIdentity();
		GLU.gluOrtho2D(gl, 0.0f, 1.0f, 0.0f, 1.0f);
		
		gl.glEnable(GL10.GL_DEPTH_TEST);
		gl.glDepthFunc(GL10.GL_LEQUAL);
		
		gl.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		gl.glClearDepthf(1.0f);
		
		gl.glEnable(GL10.GL_CULL_FACE);
		gl.glShadeModel(GL10.GL_SMOOTH);
			
	}

	// cuando se crea la superficie para pintar
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config) {
		gl.glMatrixMode(GL10.GL_PROJECTION);
		gl.glLoadIdentity();
		GLU.gluOrtho2D(gl, 0.0f, 1.0f, 0.0f, 1.0f);
		
		gl.glEnable(GL10.GL_DEPTH_TEST);
		gl.glDepthFunc(GL10.GL_LEQUAL);
		
		gl.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		gl.glClearDepthf(1.0f);
		
		gl.glEnable(GL10.GL_CULL_FACE);
		gl.glShadeModel(GL10.GL_SMOOTH);		
	}



}
