package com.example.project;


import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;
import com.google.gson.Gson;
import com.googlecode.tesseract.android.TessBaseAPI;
import com.mannan.translateapi.Language;
import com.mannan.translateapi.TranslateAPI;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.sql.SQLOutput;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;


public class InceptionV3OLD extends AppCompatActivity {
    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("nonfree");
        //System.loadLibrary("nativelib");
    }
    ///////////////////////////////////////////////////////// déclaration des attributs ////////////////////////////////////////////////////
    private static final String TAG = "MainActivity";


    private TensorFlowInferenceInterface inferenceInterface;
    //afficher la photo dans l'ImageView
    private ImageView imgview;
    private Button picture,predict,first,GENERATE_CAPTION,GENERATE_CAPTION1;
    private TextView tv;
    private TextView tv1;
    private TextView tv2;
    private Bitmap img;


    private ProgressDialog dialog;
    TextToSpeech t1;
    Button translateButton;
    Button ocr;
    Button speech;


    private TessBaseAPI mTess;
    String datapath = "";
    private FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
    private static final int PICTURE_RESULT = 9;
    private static final String MODEL_FILE = "file:///android_asset/inceptionV3/optimized_InceptionV3_RNN.pb";
    private static final String INPUT1 = "encoder/import/input_1:0";
    private static final String OUTPUT_NODES = "inceptionV3/DecoderOutputs_RNN.txt";
    private TensorFlowInferenceInterface InferenceInterface;
    private static final int[] DIM_IMAGE = new int[]{1, 299, 299, 3};
    private static final int IMAGE_SIZE = 299;
    private static final int NUM_TIMESTEPS = 22;
    private String[] OutputNodes = null;
    private String[] WORD_MAP = null;
    private static final int PICK_IMAGE_REQUEST = 1;
    Uri imageUri;
    private static final String API_KEY = "MY_API_KEY";
    ///////////////////////////////////////////////////////// méthode  ////////////////////////////////////////////////////
    public void onDestroy() {
        if (img != null) {
            img.recycle();
        }
        super.onDestroy();
    }
    @Override
    public void onResume() {
        super.onResume();

    }
    ///////////////////////////////////////////// language supported for Translation///////////////////////////////////////////////////
    private void ChangeLanguageDialog(){
        final String[] listItems = {"French","Spanish","Italian","Arabic","Hindi"};
        AlertDialog.Builder builder = new AlertDialog.Builder(InceptionV3OLD.this);
        builder.setTitle("choose language ....");
        builder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int k) {
                inferenceInterface = InitSession();
                final String text = runModel(img);
                //final String text = res;
                if (k == 0) {
                    TranslateAPI translateAPI = new TranslateAPI(
                            Language.AUTO_DETECT,
                            Language.FRENCH,text);
                    translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                        @Override
                        public void onSuccess(String translatedText) {
                            Log.d(TAG, "onSuccess: " + translatedText);
                            tv1.setText(translatedText);
                        }
                        @Override
                        public void onFailure(String ErrorText) {
                            Log.d(TAG, "onFailure: " + ErrorText);
                        }
                    });
                }
                if (k == 1) {
                    TranslateAPI translateAPI = new TranslateAPI(
                            Language.AUTO_DETECT,
                            Language.SPANISH, text);
                    translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                        @Override
                        public void onSuccess(String translatedText) {
                            Log.d(TAG, "onSuccess: " + translatedText);
                            tv1.setText(translatedText);
                        }
                        @Override
                        public void onFailure(String ErrorText) {
                            Log.d(TAG, "onFailure: " + ErrorText);
                        }
                    });
                }
                if (k == 2) {
                    TranslateAPI translateAPI = new TranslateAPI(
                            Language.AUTO_DETECT,
                            Language.ITALIAN, text);
                    translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                        @Override
                        public void onSuccess(String translatedText) {
                            Log.d(TAG, "onSuccess: " + translatedText);
                            tv1.setText(translatedText);
                        }
                        @Override
                        public void onFailure(String ErrorText) {
                            Log.d(TAG, "onFailure: " + ErrorText);
                        }
                    });
                }
                if (k == 3) {
                    TranslateAPI translateAPI = new TranslateAPI(
                            Language.AUTO_DETECT,
                            Language.ARABIC, text);
                    translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                        @Override
                        public void onSuccess(String translatedText) {
                            Log.d(TAG, "onSuccess: " + translatedText);
                            tv1.setText(translatedText);
                        }
                        @Override
                        public void onFailure(String ErrorText) {
                            Log.d(TAG, "onFailure: " + ErrorText);
                        }
                    });
                }
                if (k == 4) {
                    TranslateAPI translateAPI = new TranslateAPI(
                            Language.AUTO_DETECT,
                            Language.HINDI, text);
                    translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                        @Override
                        public void onSuccess(String translatedText) {
                            Log.d(TAG, "onSuccess: " + translatedText);
                            tv1.setText(translatedText);
                        }
                        @Override
                        public void onFailure(String ErrorText) {
                            Log.d(TAG, "onFailure: " + ErrorText);
                        }
                    });
                }
                dialogInterface.dismiss();
            }
        });
        AlertDialog dialog = builder.create();
        dialog.show();
    }
    ///////////////////////////////////////////// language supported for OCR///////////////////////////////////////////////////
    private void ChangeLanguageDialoggOCR(){
        final String[] listItems = {"French","Spanish","Italian","Arabic","Hindi"};
        AlertDialog.Builder builder = new AlertDialog.Builder(InceptionV3OLD.this);
        builder.setTitle("choose language ....");
        builder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int k) {
                TextRecognizer recognizer = new TextRecognizer.Builder(getApplicationContext()).build();
                Frame frame = new Frame.Builder().setBitmap(img).build();
                SparseArray<TextBlock> items = recognizer.detect(frame);
                if (items.size() == 0) {
                    Toast.makeText(InceptionV3OLD.this, "No Text Detected", Toast.LENGTH_SHORT).show();
                    final String text1 = "No Text Detected";
                    if (k == 0) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.FRENCH, text1);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 1) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.SPANISH, text1);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 2) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.ITALIAN, text1);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }
                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 3) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.ARABIC, text1);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 4) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.HINDI, text1);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                } else {
                    StringBuilder sb = new StringBuilder();
                    //get text from sb until there is no text
                    for (int i = 0; i < items.size(); i++) {
                        TextBlock myItem = items.valueAt(i);
                        sb.append(myItem.getValue());
                        sb.append("\n");
                    }
                    final String text = sb.toString();
                    if (k == 0) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.FRENCH, text);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 1) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.SPANISH, text);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 2) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.ITALIAN, text);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 3) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.ARABIC, text);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    if (k == 4) {
                        TranslateAPI translateAPI = new TranslateAPI(
                                Language.AUTO_DETECT,
                                Language.HINDI, text);
                        translateAPI.setTranslateListener(new TranslateAPI.TranslateListener() {
                            @Override
                            public void onSuccess(String translatedText) {
                                Log.d(TAG, "onSuccess: " + translatedText);
                                tv2.setText(translatedText);
                            }

                            @Override
                            public void onFailure(String ErrorText) {
                                Log.d(TAG, "onFailure: " + ErrorText);
                            }
                        });
                    }
                    dialogInterface.dismiss();
                }
            }
        });
        AlertDialog dialog = builder.create();
        dialog.show();
    }
    ///////////////////////////////////////////// méthode caption image//////////////////////////////////////////////////
    // lire les contenu dans les fichiers asset
    String[] LoadFile(String fileName){
        InputStream is = null;
        try {
            is = this.getAssets().open(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        BufferedReader r = new BufferedReader(new InputStreamReader(is));
        StringBuilder total = new StringBuilder();
        String line;
        try {
            while ((line = r.readLine()) != null) {
                total.append(line).append('\n');
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return total.toString().split("\n");

    }
    TensorFlowInferenceInterface InitSession(){
        InferenceInterface = new TensorFlowInferenceInterface(this.getAssets(), MODEL_FILE);
        OutputNodes = LoadFile(OUTPUT_NODES);
        WORD_MAP = LoadFile("IdmapInceptionV3_RNN");
        Log.d("DEBUG","INIT SESSION");
        return InferenceInterface;
    }
    String runModel(Bitmap bitmap){
        return GenerateCaptions(Preprocess(bitmap));
    }
    float[] Preprocess(Bitmap bitmap) {
        bitmap = Bitmap.createScaledBitmap(bitmap, 299, 299, true);

        int[] intValues = new int[299 * 299];
        float[] floatValues = new float[299 * 299 * 3];

        bitmap.getPixels(intValues, 0, 299, 0, 0, 299, 299);

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((float) ((val >> 16) & 0xFF)) / 255;//R
            floatValues[i * 3 + 1] = ((float) ((val >> 8) & 0xFF)) / 255;//G
            floatValues[i * 3 + 2] = ((float) ((val & 0xFF))) / 255;//B
        }
        return floatValues;
    }
    String GenerateCaptions(float[] imRGBMatrix){
        long startTime = SystemClock.currentThreadTimeMillis();
        InferenceInterface.feed(INPUT1, imRGBMatrix, 1, 299,299, 3);
        InferenceInterface.run(OutputNodes);
        String result = "";
        int temp[][]= new int[22][1];
        for(int i = 0; i<22; ++i) {
            InferenceInterface.fetch(OutputNodes[i], temp[i]);

            if(temp[i][0] == 2/*</S>*/){
                long costTime = SystemClock.currentThreadTimeMillis() - startTime;
                Log.i("GenerateCaptions", "GenerateCaptions end, cost time=" + costTime + "ms");
                return result;
            }

            result += WORD_MAP[temp[i][0]]+" ";
        }
        return null;
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
    // https://github.com/tesseract-ocr/tessdata
    // https://tesseract-ocr.github.io/tessdoc/Data-Files.html
    public void AppelDetecetionText() {

    }

    ////////////////////////////////////////////////////////les buttons Interfaces /////////////////////////////////////////////
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //inference = new Inference(getAssets());
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
                dialog = ProgressDialog.show(InceptionV3OLD.this, "I am captioning...", "please wait..", true);
                if (OpenCVLoader.initDebug()) {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera.putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT);
                    //Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
                    //Log.i(TAG, "******************************Opencv is added successfullu!");
                    Toast.makeText(InceptionV3OLD.this, "OpenCV Loaded Successfully!", Toast.LENGTH_SHORT).show();
                }else {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera  = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera .putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT );
                    //Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
                    //Log.i(TAG, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^Opencv is not working properly!!");
                    Toast.makeText(InceptionV3OLD.this, "OpenCV Failed!", Toast.LENGTH_SHORT).show();
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
                Toast.makeText(InceptionV3OLD.this, "Select Picture", Toast.LENGTH_SHORT).show();
            }
        });
        ////////////////////////////////////////////////OCR/////////////////////////////////////////////////////
        ocr.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ChangeLanguageDialoggOCR();
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
                inferenceInterface = InitSession();
                final String text = runModel(img);
                tv1.setText(text);
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
