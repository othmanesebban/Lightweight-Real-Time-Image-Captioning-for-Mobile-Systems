package com.example.project;


import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Build;
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

import com.google.android.gms.vision.Detector;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;
import com.google.gson.Gson;
import com.mannan.translateapi.Language;
import com.mannan.translateapi.TranslateAPI;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

import org.opencv.core.Point;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;

import org.opencv.features2d.KeyPoint;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/*   Inception-v3 */
public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("nonfree");
        //System.loadLibrary("native-lib");
    }
    private static final String TAG = "MainActivity";
    //afficher la photo dans l'ImageView
    private ImageView imgview;
    private Button picture,predict,first,GENERATE_CAPTION,GENERATE_CAPTION1;
    private TextView tv;
    private TextView tv1;
    private TextView tv2;
    private Bitmap img;
    Interpreter interpreter;
    Interpreter encoder;
    Interpreter decoder;
    private ProgressDialog dialog;
    private TextToSpeech t1;
    private Button translateButton;
    private Button ocr;
    private Button speech;
    private static int selectedLanguage = 1;
    Locale french = new Locale("fr","FR");
    private FeatureDetector detectorSIFT = FeatureDetector.create(FeatureDetector.SIFT);
    private FeatureDetector detectorSURF = FeatureDetector.create(FeatureDetector.SURF);


    // Créer un détecteur de caractéristiques BRISK
    FeatureDetector detectorBRISK = FeatureDetector.create(FeatureDetector.BRISK);
    private FeatureDetector detectorORB = FeatureDetector.create(FeatureDetector.ORB); // New line for ORB detector

    //private SIFT detector = SIFT.create();
    private static final int PICTURE_RESULT = 9;
    //private static final String MODEL_FILE = "file:///android_asset/optimized_InceptionV3_RNN.pb";
    //private static final String INPUT1 = "encoder/import/input_1:0";
    //private static final String OUTPUT_NODES = "DecoderOutputs.txt";
    private TensorFlowInferenceInterface InferenceInterface;
    private static final int[] DIM_IMAGE = new int[]{1, 299, 299, 3};
    private String[] OutputNodes = null;
    private String[] WORD_MAP = null;
    private static final int PICK_IMAGE_REQUEST = 1;
    Uri imageUri;
    private static final String API_KEY = "MY_API_KEY";


    public void onDestroy() {
        if (img != null) {
            img.recycle();
        }
        if (t1 != null) {
            t1.stop();
            t1.shutdown();
        }
        super.onDestroy();
    }
    /*public void onPause(){
        if(t1 !=null){
            t1.stop();
            t1.shutdown();
        }
        super.onPause();
    }*/
    @Override
    public void onResume() {
        super.onResume();

    }
    //////////////////////////////////language supported for Translation////////////////////////////
    private void ChangeLanguageDialog(){
        final String[] listItems = {"French","Spanish","Italian","Arabic","Hindi"};
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("choose language ....");
        builder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int k) {
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
                final String text = res;
                TranslateAPI translateAPI;
                switch (k) {
                    case 0: // French
                        translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.FRENCH, text);
                        break;
                    case 1: // Spanish
                        translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.SPANISH, text);
                        break;
                    case 2: // Italian
                        translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.ITALIAN, text);
                        break;
                    case 3: // Arabic
                        translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.ARABIC, text);
                        break;
                    case 4: // Hindi
                        translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.HINDI, text);
                        break;
                    default:
                        translateAPI = null; // ou gérer autrement si l'indice est inattendu
                        break;
                }
                if (translateAPI != null) {
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
    ////////////////////////// language supported for OCR/////////////////////////////////////
    private void ChangeLanguageDialogOCR(){
        final String[] listItems = {"French","Spanish","Italian","Arabic","Hindi"};
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("choose language ....");
        builder.setSingleChoiceItems(listItems, -1, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int k) {
                TextRecognizer recognizer = new TextRecognizer.Builder(getApplicationContext()).build();
                Frame frame = new Frame.Builder().setBitmap(img).build();
                SparseArray<TextBlock> items = recognizer.detect(frame);
                if (items.size() == 0) {
                    //Toast.makeText(MainActivity.this, "No Text Detected", Toast.LENGTH_SHORT).show();
                    final String text1 = "No Text Detected";
                    TranslateAPI translateAPI;
                    switch (k) {
                        case 0: // French
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.FRENCH, text1);
                            break;
                        case 1: // Spanish
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.SPANISH, text1);
                            break;
                        case 2: // Italian
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.ITALIAN, text1);
                            break;
                        case 3: // Arabic
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.ARABIC, text1);
                            break;
                        case 4: // Hindi
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.HINDI, text1);
                            break;
                        default:
                            translateAPI = null; // ou gérer autrement si l'indice est inattendu
                            break;
                    }
                    if (translateAPI != null) {
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
                    TranslateAPI translateAPI;
                    switch (k) {
                        case 0: // French
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.FRENCH, text);
                            break;
                        case 1: // Spanish
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.SPANISH, text);
                            break;
                        case 2: // Italian
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.ITALIAN, text);
                            break;
                        case 3: // Arabic
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.ARABIC, text);
                            break;
                        case 4: // Hindi
                            translateAPI = new TranslateAPI(Language.AUTO_DETECT, Language.HINDI, text);
                            break;
                        default:
                            translateAPI = null; // ou gérer autrement si l'indice est inattendu
                            break;
                    }
                    if (translateAPI != null) {
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
    ////////////////////////////////////////////////////////la méthode SIFT & SURF/////////////////////////////////////////////
    public void extractFeaturesUsingDetector(final FeatureDetector detector, final DescriptorExtractor extractor) {
        /////////////////////////////////////// 1 ére méthode ////////////////////////////////////////////
       new Thread(new Runnable() {
            @Override
            public void run() {
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
       }).start();
    }


        /////////////////////////////////////// 2 éme méthode ////////////////////////////////////////////

        /*new Thread(new Runnable() {
            @Override
            public void run() {
                // Initialisation du temps de départ
                final long startTime = SystemClock.currentThreadTimeMillis();
                Mat rgba = new Mat();
                Utils.bitmapToMat(img, rgba);
                // Temps après conversion en Mat
                final long afterMatConversion = SystemClock.currentThreadTimeMillis();
                // Conversion de l'image en niveaux de gris
                Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_BGR2GRAY);
                // Temps après conversion en niveaux de gris
                final long afterGrayConversion = SystemClock.currentThreadTimeMillis();
                // Détection des points clés
                MatOfKeyPoint keyPoints = new MatOfKeyPoint();
                detector.detect(rgba, keyPoints);
                // Temps après détection des points clés
                final long afterKeyPointsDetection = SystemClock.currentThreadTimeMillis();
                // Liste des points détectés
                List<KeyPoint> detectedKeypoints = keyPoints.toList();
                List<Point> detectedPoints = new ArrayList<>();
                for (KeyPoint kp : detectedKeypoints) {
                    detectedPoints.add(kp.pt);
                }
                // Liste des points de vérité terrain
                List<Point> groundTruthPoints = getGroundTruthPoints();
                // Calcul des correspondances correctes
                int correctMatches = 0;
                double tolerance = 20.0; // Tolérance pour les correspondances correctes
                for (Point detectedPoint : detectedPoints) {
                    for (Point truthPoint : groundTruthPoints) {
                        if (Math.abs(detectedPoint.x - truthPoint.x) < tolerance &&
                                Math.abs(detectedPoint.y - truthPoint.y) < tolerance) {
                            correctMatches++;
                            break; // Sortir de la boucle si une correspondance est trouvée
                        }
                    }
                }
                // Temps après calcul des correspondances
                final long afterMatchCalculation = SystemClock.currentThreadTimeMillis();
                // Calcul de la précision
                int totalDetected = detectedPoints.size();
                double precision = totalDetected > 0 ? (double) correctMatches / totalDetected : 0.0;
                // Dessin des points clés sur l'image pour affichage
                Features2d.drawKeypoints(rgba, keyPoints, rgba);
                // Conversion du Mat en Bitmap pour affichage
                Utils.matToBitmap(rgba, img);
                final long endTime = SystemClock.currentThreadTimeMillis();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // Affichage du temps écoulé et de la précision
                        tv.setText("Temps total: " + (endTime - startTime) + " ms\n" +
                                "Conversion Mat: " + (afterMatConversion - startTime) + " ms\n" +
                                "Conversion Gray: " + (afterGrayConversion - afterMatConversion) + " ms\n" +
                                "Détection Points: " + (afterKeyPointsDetection - afterGrayConversion) + " ms\n" +
                                "Calcul Correspondances: " + (afterMatchCalculation - afterKeyPointsDetection) + " ms\n" +
                                "Précision: " + String.format("%.2f", precision * 100) + "%");

                        imgview.setImageBitmap(img);
                    }
                });
                // Vérification des correspondances
                Log.d("DEBUG", "Total Detected: " + totalDetected);
                Log.d("DEBUG", "Correct Matches: " + correctMatches);
                Log.d("DEBUG", "Précision: " + (precision * 100) + "%");
            }
        }).start();
    }*/
    public void sift() {
        extractFeaturesUsingDetector(FeatureDetector.create(FeatureDetector.SIFT), DescriptorExtractor.create(DescriptorExtractor.SIFT));
    }
    public void surf() {
        extractFeaturesUsingDetector(FeatureDetector.create(FeatureDetector.SURF), DescriptorExtractor.create(DescriptorExtractor.SURF));
    }
    public void brisk() {
        extractFeaturesUsingDetector(FeatureDetector.create(FeatureDetector.BRISK), DescriptorExtractor.create(DescriptorExtractor.BRISK));
    }
    public void orb() {
        extractFeaturesUsingDetector(FeatureDetector.create(FeatureDetector.ORB), DescriptorExtractor.create(DescriptorExtractor.ORB));
    }
    //////////////////////////////// calculer précision des algo SIFT SURF ORB BRISK /////////////////////
    // Fonction pour obtenir la vérité terrain des points
    public List<Point> getGroundTruthPoints() {
        List<Point> groundTruth = new ArrayList<>();
        // Ajoutez les points de vérité terrain ici, par exemple :
        groundTruth.add(new Point(100, 150));
        groundTruth.add(new Point(200, 250));
        groundTruth.add(new Point(300, 350));
        // Ajouter plus de points selon vos besoins
        return groundTruth;
    }


    ////////////////////////////////////////////////////////la méthode OCR/////////////////////////////////////////////
    // Google Mobile Vision API
    public void AppelDetecetionText() {
        /************************************ TextRecognitionOCR *******************************************/
        TextRecognizer recognizer = new TextRecognizer.Builder(getApplicationContext()).build();
        if (!recognizer.isOperational()) {
            Toast.makeText(MainActivity.this, "Error", Toast.LENGTH_SHORT).show();
        } else {
            Frame frame = new Frame.Builder().setBitmap(img).build();
            SparseArray<TextBlock> items = recognizer.detect(frame);
            if (items.size() == 0) {
                Toast.makeText(MainActivity.this, "No Text Detected", Toast.LENGTH_SHORT).show();
                tv2.setText("No Text Detected");
            } else {
                StringBuilder sb = new StringBuilder();
                //get text from sb until there is no text
                for (int i = 0; i < items.size(); i++) {
                    TextBlock myItem = items.valueAt(i);
                    sb.append(myItem.getValue());
                    sb.append("\n");
                }
                tv2.setText(sb.toString());
            }
        }
        /*
        //////////////////////////////////// 2 éme méthode /////////////////////
        Frame frame = new Frame.Builder().setBitmap(img).build();
        Detector<TextBlock> detections = new Detector<TextBlock>() {
            @Override
            public SparseArray<TextBlock> detect(Frame frame) {
                SparseArray<TextBlock> items = recognizer.detect(frame);
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < items.size(); ++i) {
                    TextBlock item = items.valueAt(i);
                    if (item != null && item.getValue() != null) {
                        Log.d("OcrDetectorProcessor", "Text detected! " + item.getValue());
                        sb.append(item.getValue());
                        sb.append("\n");
                    }
                }
                // Mettre à jour l'interface utilisateur avec le texte détecté
                runOnUiThread(() -> tv2.setText(sb.toString()));
                return items;
            }
        };
        // Utilisez votre détecteur ici sur le frame.
        // Notez que ce code s'exécute déjà dans le contexte de votre détecteur personnalisé.
        SparseArray<TextBlock> items = detections.detect(frame);
        // Le traitement du résultat peut continuer ici si nécessaire.*/
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
                dialog = ProgressDialog.show(MainActivity.this, "I am captioning...", "please wait..", true);
                if (OpenCVLoader.initDebug()) {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera.putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT);
                    //Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
                    //Log.i(TAG, "******************************Opencv is added successfullu!");
                    Toast.makeText(MainActivity.this, "OpenCV Loaded Successfully!", Toast.LENGTH_SHORT).show();
                }else {
//                    Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                    Intent camera  = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                    camera .putExtra("android.intent.extra.quickCapture", true);
                    startActivityForResult(camera, PICTURE_RESULT );
                    //Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
                    //Log.i(TAG, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^Opencv is not working properly!!");
                    Toast.makeText(MainActivity.this, "OpenCV Failed!", Toast.LENGTH_SHORT).show();
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
                Toast.makeText(MainActivity.this, "Select Picture", Toast.LENGTH_SHORT).show();
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

                speak();
            }
        });
    }
    private void speak() {
        //https://github.com/himasaik/Android-Text-to-Speech-Converter/blob/master/LHD-2019-Incomplete-90207651a0ef7a5b2a36ba15d57a33b0cc0fce72/app/src/main/java/com/example/lhd_2019/MainActivity.java
        String toSpeak = tv1.getText().toString();
        t1.setPitch(0.6f); // Set pitch level
        t1.setSpeechRate(1.0f); // Set speed rate
        t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null, null);
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
                //brisk();
                //surf();
                //sift();
                //orb();
                // on appelle pour la méthode captioning ici
                //caption_image();
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