package com.example.project;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.net.Uri;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import com.google.gson.Gson;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import android.content.res.AssetManager;
/*   Inception-v3 */
public class TesseractWess extends AppCompatActivity {
    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("nonfree");
        //System.loadLibrary("nativelib");
    }
    private static final String TAG = "MainActivity";
    //afficher la photo dans l'ImageView
    private ImageView imgview;
    private Button picture,predict,first,GENERATE_CAPTION,GENERATE_CAPTION1;
    String language = "eng";
    private TessBaseAPI mTess;
    String datapath = "";
    private TextView tv;
    private TextView tv1;
    private TextView tv2;
    private Bitmap img;
    Interpreter interpreter;
    Interpreter encoder;
    Interpreter decoder;
    private ProgressDialog dialog;
    TextToSpeech t1;
    Button translateButton;
    Button ocr;
    Button speech;
    private FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
    private static final int PICTURE_RESULT = 9;
    private static final int PICK_IMAGE_REQUEST = 1;
    Uri imageUri;

    public void onDestroy() {
        if (img != null) {img.recycle();}
        if (mTess != null) mTess.end();
        super.onDestroy();
    }
    public void onPause(){
        if(t1 !=null){
            t1.stop();
            t1.shutdown();
        }
        super.onPause();
    }
    @Override
    public void onResume() {super.onResume();}
    ///////////////////////////////////////////// language supported for Translation///////////////////////////////////////////////////
    private void ChangeLanguageDialog(){

    }
    ///////////////////////////////////////////// language supported for OCR///////////////////////////////////////////////////
    private void ChangeLanguageDialogOCR(){

    }
    ///////////////////////////////////////////// méthode caption image//////////////////////////////////////////////////
    // lire les contenu dans les fichiers asset
    public void caption_image(){
        long startTime = SystemClock.currentThreadTimeMillis();
        HashMap<String, Integer> wordToIndices=new HashMap<>();
        HashMap<Integer, String> indexToWords=new HashMap<>();
        String json;
        try {
            InputStream wi = getAssets().open("sample.json");
            int size = wi.available();
            byte[] buffer = new byte[size];
            wi.read(buffer);
            wi.close();
            json = new String(buffer, "UTF-8");
            Gson gson=new Gson();
            wordToIndices = gson.fromJson(json, HashMap.class);

        } catch (IOException ex) {
            ex.printStackTrace();
        }
        try {
            InputStream iw = getAssets().open("index_to_word.json");
            int size = iw.available();
            byte[] buffer = new byte[size];
            iw.read(buffer);
            iw.close();
            json = new String(buffer, "UTF-8");
            Gson gson=new Gson();
            indexToWords = gson.fromJson(json, HashMap.class);
//                        tv.setText(indexToWords.get("2341"));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        try {
            interpreter= new Interpreter(loadModelFile(),null);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = Bitmap.createScaledBitmap(img, 299, 299, true);
        ByteBuffer input = ByteBuffer.allocateDirect(299 * 299 * 3 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer output=ByteBuffer.allocateDirect(1*8*8*2048*4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < 299; y++) {
            for (int x = 0; x < 299; x++) {
                int px = bitmap.getPixel(x, y);
                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);
                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = (r - 127) / 255.0f;
                float gf = (g - 127) / 255.0f;
                float bf = (b - 127) / 255.0f;
                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }
        input.rewind();
        interpreter.run(input,output);
        output.rewind();
        float[][][] encoutput=new float[1][64][256];
        try {
            encoder=new Interpreter(loadencoder(),null);
        } catch (IOException e) {
            e.printStackTrace();
        }
        encoder.resizeInput(0,new int[]{1,64,2048});
        encoder.run(output,encoutput);
        try {
            decoder=new Interpreter(loaddecoder(),null);
        } catch (IOException e) {
            e.printStackTrace();
        }
        int[][] decInput=new int[1][1];
        float[][] hidden=new float[1][512];
        String res = " ";
        decInput[0][0]=3;
        for(int j=0;j<52;j++)
        {
            Object[] inputs={decInput,encoutput,hidden};
            float[][] pred=new float[1][5001];
            float[][][] attn=new float[1][64][1];

            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, pred);
            outputs.put(1, hidden);
            outputs.put(2, attn);
            decoder.runForMultipleInputsOutputs(inputs, outputs);
            float large=0.0f;
            int predIdx= -1;
            for(int i=0;i<pred[0].length;i++)
            {
                if(pred[0][i]>large){
                    large=pred[0][i];
                    predIdx=i;
                }
            }
            if(predIdx==4)
                break;
            decInput[0][0]=predIdx;
            int temp[][]= new int[22][1];
            for(int i = 0; i<22; ++i){
                if(temp[i][0] == 10/*</S>*/){

                    long costTime = SystemClock.currentThreadTimeMillis() - startTime;
                    Log.i("GenerateCaptions", "GenerateCaptions end, cost time=" + costTime + "ms");
                    res+=temp[i][0]+" "+indexToWords.get(String.valueOf(predIdx));
                }
            }
            res+=" "+indexToWords.get(String.valueOf(predIdx));
        }
        tv1.setText(res);
        imgview.setImageBitmap(img);
        //t1.speak(res, TextToSpeech.QUEUE_FLUSH, null);
        interpreter.close();
        encoder.close();
        decoder.close();
    }
    ////////////////////////////////////////////////////////la méthode SIFT/////////////////////////////////////////////
    public void sift() {
        new Thread(){
            @Override
            public void run() {
                super.run();
                final long startTime = SystemClock.currentThreadTimeMillis();
                Mat rgba = new Mat();
                Utils.bitmapToMat(img, rgba);
                MatOfKeyPoint keyPoints = new MatOfKeyPoint();
                Imgproc.cvtColor(rgba, rgba, Imgproc.INTER_MAX);
                detector.detect(rgba, keyPoints);
                Features2d.drawKeypoints(rgba, keyPoints, rgba);
                Utils.matToBitmap(rgba, img);
                final long endTime = SystemClock.currentThreadTimeMillis();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tv.setText(endTime-startTime+"");
                        imgview.setImageBitmap(img);
                    }
                });
            }
        }.start();
    }
    ////////////////////////////////////////////////////////la méthode OCR/////////////////////////////////////////////
    //
    //
    public void AppelDetecetionText() {
        String OCRresult = null;
        //if(OCRresult!=null) {
        mTess.setImage(img);
        OCRresult = mTess.getUTF8Text();
        Toast.makeText(getBaseContext(), OCRresult, Toast.LENGTH_SHORT).show();
        tv2.setText(OCRresult);
       /* }else{
            tv2.setText("No Text Detected");
        }*/
    }
    private void checkFile(File dir) {
        if (!dir.exists()&& dir.mkdirs()){
            System.out.println("nothing exists");
        }
        if(dir.exists()) {
            String[] languages = language.split("\\+");
            for (String lang : languages) {
                String datafilepath = datapath + "/tessdata/" + lang + ".traineddata";
                File datafile = new File(datafilepath);
                if (!datafile.exists()) {
                    copyFiles(lang);
                }
            }
        }
    }
    private void copyFiles(String lang) {
        try {
            String filename = lang + ".traineddata";
            String filepath = datapath + "/tessdata/"+filename;
            AssetManager assetManager = getAssets();
            InputStream instream = assetManager.open("/tessdata/"+filename);
            OutputStream outstream = new FileOutputStream(filepath);
            byte[] buffer = new byte[1024];
            int read;
            while ((read = instream.read(buffer)) != -1) {
                outstream.write(buffer, 0, read);
            }
            outstream.flush();
            outstream.close();
            instream.close();
            File file = new File(filepath);
            if (!file.exists()) {
                throw new FileNotFoundException();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgview = (ImageView)findViewById(R.id.imageView);
        GENERATE_CAPTION=(Button)findViewById(R.id.button);
        GENERATE_CAPTION1=(Button)findViewById(R.id.button6);
        predict=(Button)findViewById(R.id.button2);
        tv=(TextView)findViewById(R.id.textView2);
        tv1=(TextView)findViewById(R.id.textView3);
        tv2=(TextView)findViewById(R.id.textView4);
        first = findViewById(R.id.first);
        translateButton = findViewById(R.id.button3);
        ocr = findViewById(R.id.button4);
        speech = findViewById(R.id.button5);
        ////////////////////////////////// ocr ////////////////////////////
        datapath = getFilesDir()+ "/tesseract/";
        mTess = new TessBaseAPI();
        checkFile(new File(datapath + "tessdata/"));
        mTess.init(datapath, language);
        /////////////////////////////////////////////text to speech///////////////////////////////////////////
        t1=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener()
        {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int result = t1.setLanguage(Locale.ENGLISH);
                    if (result == TextToSpeech.LANG_MISSING_DATA
                            || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Toast.makeText(getApplicationContext(), "This language is not supported", Toast.LENGTH_SHORT).show();
                        //showToast("This language is not supported");
                    }
                    else{
                        Log.v("TTS","onInit succeeded");
                        //Lspeak("working");
                    }
                } else {
                    Toast.makeText(getApplicationContext(), "Initialization failed", Toast.LENGTH_SHORT).show();
                    //showToast("Initialization failed");
                }
            }
        });
        ////////////////////////////////////////////////prend image/////////////////////////////////////////////////////
        GENERATE_CAPTION.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dialog = ProgressDialog.show(TesseractWess.this, "I am captioning...", "please wait..", true);
                if (OpenCVLoader.initDebug()) {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera.putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT);
                    //Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
                    //Log.i(TAG, "******************************Opencv is added successfullu!");
                    Toast.makeText(TesseractWess.this, "OpenCV Loaded Successfully!", Toast.LENGTH_SHORT).show();
                }else {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera  = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera .putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT );
                    //Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
                    //Log.i(TAG, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^Opencv is not working properly!!");
                    Toast.makeText(TesseractWess.this, "OpenCV Failed!", Toast.LENGTH_SHORT).show();
                }
            }
        });
        ////////////////////////////////////////////////prend image/////////////////////////////////////////////////////
        GENERATE_CAPTION1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);//
                //startActivityForResult(Intent.createChooser(intent, "Select File"), PICK_IMAGE_REQUEST);
                startActivityForResult(intent,PICK_IMAGE_REQUEST);
                Toast.makeText(TesseractWess.this, "Select Picture", Toast.LENGTH_SHORT).show();
            }
        });
        ////////////////////////////////////////////////OCR/////////////////////////////////////////////////////
        ocr.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                ChangeLanguageDialogOCR();
            }
        });
        ////////////////////////////////////////////////translation/////////////////////////////////////////////////////
        translateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ChangeLanguageDialog();
            }
        });
        ////////////////////////////////////////////////speech/////////////////////////////////////////////////////
        speech.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String toSpeak = tv1.getText().toString();
                t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null);
            }
        });
    }
    private MappedByteBuffer loadModelFile() throws IOException
    {
        AssetFileDescriptor assetFileDescriptor=this.getAssets().openFd("feature.tflite");
        FileInputStream fileInputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=fileInputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long length=assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,length);
    }
    private MappedByteBuffer loadencoder() throws IOException
    {
        AssetFileDescriptor assetFileDescriptor=this.getAssets().openFd("encoder.tflite");
        FileInputStream fileInputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=fileInputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long length=assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,length);
    }
    private MappedByteBuffer loaddecoder() throws IOException
    {
        AssetFileDescriptor assetFileDescriptor=this.getAssets().openFd("decoder.tflite");
        FileInputStream fileInputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=fileInputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long length=assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,length);
    }

    /*void speak(String s){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            Log.v("TAG", "Speak new API");
            Bundle bundle = new Bundle();
            bundle.putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_MUSIC);
            t1.speak(s, TextToSpeech.QUEUE_FLUSH, bundle, null);
        } else {
            Log.v("TAG", "Speak old API");
            HashMap<String, String> param = new HashMap<>();
            param.put(TextToSpeech.Engine.KEY_PARAM_STREAM, String.valueOf(AudioManager.STREAM_MUSIC));
            t1.speak(s, TextToSpeech.QUEUE_FLUSH, param);
        }
    }*/
    //récupère la photo prise renvoyée en retour de la caméra.
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //dialog = ProgressDialog.show(MainActivity4.this, "I am captioning...", "please wait..", true);
        // si les résultats proviennent de l'activité de la caméra
        if (requestCode == PICTURE_RESULT) {
            // si une photo a été prise
            if (resultCode == Activity.RESULT_OK) {
                // Libérer les données de la dernière image
                if (img != null)
                    img.recycle();
                // Obtenir la photo prise par l'utilisateur
                img = (Bitmap) data.getExtras().get("data");
                // Éviter l'IllegalStateException avec un bitmap immuable
                Bitmap pic = img.copy(img.getConfig(), true);
                img.recycle();
                img = pic;
                // Montrer l'image
                imgview.setImageBitmap(img);
                // on appelle pour la méthode sift ici
                //sift();
                // on appelle pour la méthode captioning ici
                caption_image();
                // on appelle pour la méthode detection du texte ici
                AppelDetecetionText();
                if (dialog != null) {
                    dialog.dismiss();
                }
                // si l'utilisateur a annulé l'activité de la caméra
            } //else if (resultCode == Activity.RESULT_CANCELED) {
            //}
            if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK) {
                assert data != null;
                imageUri = data.getData();
                try {
                    img = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                    imgview.setImageBitmap(img);
                    t1.speak("Image is uploaded", TextToSpeech.QUEUE_FLUSH, null);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        }
    }


}