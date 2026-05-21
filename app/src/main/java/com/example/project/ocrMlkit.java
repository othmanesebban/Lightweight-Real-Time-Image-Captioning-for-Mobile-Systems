package com.example.project;


import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.graphics.Point;
import android.graphics.Rect;
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

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;


import com.google.gson.Gson;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/*   Inception-v3 */
public class ocrMlkit extends AppCompatActivity {
    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("nonfree");
        //System.loadLibrary("nativelib");
    }
    private static final String TAG = "MainActivity";
    //afficher la photo dans l'ImageView
    private ImageView imgview;
    Boolean check = false;
    private Button picture,predict,first,GENERATE_CAPTION,GENERATE_CAPTION1;
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
    private static int selectedLanguage = 1;
    Locale french = new Locale("fr","FR");
    private FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
    private static final int PICTURE_RESULT = 9;
    //private static final String MODEL_FILE = "file:///android_asset/optimized_InceptionV3_RNN.pb";
    //private static final String INPUT1 = "encoder/import/input_1:0";
    //private static final String OUTPUT_NODES = "DecoderOutputs.txt";
    private TensorFlowInferenceInterface InferenceInterface;
    private static final int[] DIM_IMAGE = new int[]{1, 299, 299, 3};
    private String[] OutputNodes = null;
    private String[] WORD_MAP = null;
    StringBuilder strBuilder1;
    private static final int PICK_IMAGE_REQUEST = 1;
    Uri imageUri;
    private static final String API_KEY = "MY_API_KEY";
    public void onDestroy() {
        if (img != null) {
            img.recycle();
        }
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
    public void onResume() {
        super.onResume();

    }
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
    // Google Firebase Machine Learning kit
    // https://github.com/HaiderSaleem/OCR-Android
    // https://github.com/SSAnalyst/OCR-firebase-app/tree/main
    public void AppelDetecetionText() {
        InputImage image = InputImage.fromBitmap(img, 0);
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        Task<Text> result = recognizer.process(image)
                .addOnSuccessListener(new OnSuccessListener<Text>() {
                    @Override
                    public void onSuccess(Text text) {
                        if (text != null) {
                            String recognizedText = text.getText();
                            if (!recognizedText.isEmpty()) {
                                StringBuilder result = new StringBuilder();
                                for (Text.TextBlock block : text.getTextBlocks()){
                                    String blockText = block.getText();
                                    Point[] blockCornerPoint = block.getCornerPoints();
                                    Rect blockFrame = block.getBoundingBox();
                                    for (Text.Line line : block.getLines()){
                                        String lineText = line.getText();
                                        Point[] lineCornerPoint = line.getCornerPoints();
                                        Rect lineRect = line.getBoundingBox();
                                        for (Text.Element element : line.getElements()){
                                            String elementText = element.getText();
                                            result.append(elementText);
                                        }
                                        tv2.setText(blockText);
                                    }
                                }
                            } else {
                                tv2.setText("No Text Detected");
                            }
                        } else {
                            // Text recognition result is null
                        }
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        Toast.makeText(ocrMlkit.this, "Fail To Detect Text From Image . . "+e.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
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
        /////////////////////////////////////////////text to speech///////////////////////////////////////////
        t1=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener()
        {
            @Override
            public void onInit(int status) {
                /*if (status == TextToSpeech.SUCCESS) {
                    int result = t1.setLanguage(Locale.ENGLISH);
                    if (result == TextToSpeech.LANG_MISSING_DATA ||
                            result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Toast.makeText(getApplicationContext(), "This language is not supported!",
                                Toast.LENGTH_SHORT);
                    } else {
                        speech.setEnabled(true);
                        t1.setPitch(0.6f);
                        t1.setSpeechRate(1.0f);
                    }
                }
            }*/
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
                dialog = ProgressDialog.show(ocrMlkit.this, "I am captioning...", "please wait..", true);
                if (OpenCVLoader.initDebug()) {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera.putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT);
                    //Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
                    //Log.i(TAG, "******************************Opencv is added successfullu!");
                    Toast.makeText(ocrMlkit.this, "OpenCV Loaded Successfully!", Toast.LENGTH_SHORT).show();
                }else {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera  = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera .putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT );
                    //Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
                    //Log.i(TAG, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^Opencv is not working properly!!");
                    Toast.makeText(ocrMlkit.this, "OpenCV Failed!", Toast.LENGTH_SHORT).show();
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
                Toast.makeText(ocrMlkit.this, "Select Picture", Toast.LENGTH_SHORT).show();
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