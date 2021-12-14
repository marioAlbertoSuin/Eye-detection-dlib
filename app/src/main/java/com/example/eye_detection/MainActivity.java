package com.example.eye_detection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.eye_detection.databinding.ActivityMainBinding;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private Button btnCamara;
    private ImageView imgView;
    private static String TAG = "MainActivity";
    static {
        if(OpenCVLoader.initDebug()){
            Log.d(TAG,"insalled");
        }else{
            Log.d(TAG,"not insalled");
        }
    }

    // Used to load the 'eye_detection' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        checkExternalStoragePermission();
        /* ----------------------------------------------------*/
        //String p = "/storage/emulated/0/shape_predictor_68_face_landmarks.dat";
        String p = "/storage/emulated/0/shape_predictor_68_face_landmarks.dat.bz2";

        if(new File(p).exists()){
            if (new File(p).isFile()){
                System.out.println("Es archivo");
            }else {
                System.out.println("no lo es archivo");
            }
            System.out.println("Existe");
        }else {
            System.out.println("No es existe");
        }
        /* ----------------------------------------------------*/
/*
        FileInputStream in = null;
        try {
            in = new FileInputStream("storage/emulated/0/shape_predictor_68_face_landmarks.dat.bz2");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        FileOutputStream out = null;
        try {
            out = new FileOutputStream("storage/emulated/0/shape_predictor_68_face_landmarks.dat");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BZip2CompressorInputStream bzIn = null;
        try {
            bzIn = new BZip2CompressorInputStream(in);
        } catch (IOException e) {
            e.printStackTrace();
        }
        final byte[] buffer = new byte[1024];
        int n = 0;
        while (true) {
            try {
                if (!(-1 != (n = bzIn.read(buffer)))) break;
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                out.write(buffer, 0, n);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            bzIn.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

*/
        /* ----------------------------------------------------*/
        btnCamara = findViewById(R.id.btnCamara);
        imgView = findViewById(R.id.imageView);

        btnCamara.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                abrirCamara();
            }
        });



        // Example of a call to a native method
        //TextView tv = binding.sampleText;
        //tv.setText(stringFromJNI());
    }


    private void abrirCamara(){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(intent.resolveActivity(getPackageManager()) != null){
            startActivityForResult(intent, 1);
        }
    }
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Toast.makeText(this,"Empieza proceso", Toast.LENGTH_SHORT).show();
            Bundle extras = data.getExtras();
            Bitmap imgBitmap = (Bitmap) extras.get("data");

            Mat matInput = convertBiMapMat(imgBitmap);
            Mat matoutput = new Mat (matInput.rows(),matInput.cols(),CvType.CV_8UC3);
            LandmarkDetection(matInput.getNativeObjAddr(),matoutput.getNativeObjAddr());

            imgView.setImageBitmap(convertMat2Bitmap(matoutput));
            Toast.makeText(this,"Termina proceso", Toast.LENGTH_SHORT).show();
            //imgView.setImageBitmap(convertMat2Bitmap(convertBiMapMat(imgBitmap)));
            ////////////////////////////////////
        }
    }

    // convert java Bitmap into Opencv Mat
    Mat convertBiMapMat(Bitmap rgbImage){
        Mat rgbaMat = new Mat(rgbImage.getHeight(),rgbImage.getWidth(), CvType.CV_8UC4);
        Bitmap bmp32 = rgbImage.copy(Bitmap.Config.ARGB_8888,true);
        Utils.bitmapToMat(bmp32, rgbaMat);

        Mat rgbMat = new Mat(rgbImage.getHeight(),rgbImage.getWidth(),CvType.CV_8UC3);
        Imgproc.cvtColor(rgbaMat,rgbMat,Imgproc.COLOR_RGBA2BGR,3);
        return rgbMat;
    }


    Bitmap convertMat2Bitmap(Mat img){
        int width = img.width();
        int heigth = img.height();

        Bitmap bmp;
        bmp = Bitmap.createBitmap(width,heigth,Bitmap.Config.ARGB_8888);
        Mat tmp;
        tmp = img.channels()==1? new Mat (width,heigth,CvType.CV_8UC1,new Scalar(1)): new Mat(width,heigth,CvType.CV_8UC4,new Scalar(0,0,0,0));
        try{
            if (img.channels()==3){
                Imgproc.cvtColor(img,tmp,Imgproc.COLOR_RGB2BGRA);
            }else if(img.channels()==1){
                Imgproc.cvtColor(img,tmp,Imgproc.COLOR_GRAY2RGBA);
            }
            Utils.matToBitmap(tmp,bmp);
        }catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return  bmp;
    }

    //------------- temp ------------------
    private void checkExternalStoragePermission() {
        int permissionCheck = ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE);
        if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
            Log.i("Mensaje", "No se tiene permiso para leer.");
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.READ_EXTERNAL_STORAGE}, 225);
        } else {
            Log.i("Mensaje", "Se tiene permiso para leer!");
        }
    }
    //----------------------------------------






    /**
     * A native method that is implemented by the 'eye_detection' native library,
     * which is packaged with this application.
     */
    public native void LandmarkDetection(long addrInput,long addrOutput);
}