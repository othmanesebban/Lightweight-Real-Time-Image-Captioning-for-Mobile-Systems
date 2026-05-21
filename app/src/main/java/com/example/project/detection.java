package com.example.project;

public class detection {
    /*
    public void extractFeaturesUsingDetector(final FeatureDetector detector) {
        new Thread() {
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
                        tv.setText(endTime - startTime + "");
                        imgview.setImageBitmap(img);
                    }
                });
            }
        }.start();
    }
    public void sift() {
        extractFeaturesUsingDetector(detectorSIFT);
    }
    public void surf() {
        extractFeaturesUsingDetector(detectorSURF);
    }
    public void brisk() {
        extractFeaturesUsingDetector(detectorBRISK);
    }
    // New method for ORB detection
    public void orb() { extractFeaturesUsingDetector(detectorORB); }*/
}
