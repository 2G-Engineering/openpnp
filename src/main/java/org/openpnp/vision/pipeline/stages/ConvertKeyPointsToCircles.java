package org.openpnp.vision.pipeline.stages;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.features2d.KeyPoint;
import org.openpnp.vision.pipeline.CvPipeline;
import org.openpnp.vision.pipeline.CvStage;
import org.openpnp.vision.pipeline.CvStage.Result.Circle;
import org.openpnp.vision.pipeline.CvStage.Result.TemplateMatch;
import org.openpnp.vision.pipeline.Stage;
import org.simpleframework.xml.Attribute;

@Stage(description = "Convert KeyPoints or points to circles. The center point and diameter of each will be preserved, if applicable. If the input model is a single value the result will be a single value. If the input is a List the result will be a List.")
public class ConvertKeyPointsToCircles extends CvStage {
    @Attribute(required = false)
    private String keyPointStageName;

    public String getKeyPointStageName() {
        return keyPointStageName;
    }

    public void setKeyPointStageName(String modelStageName) {
        this.keyPointStageName = modelStageName;
    }

    @Override
    public Result process(CvPipeline pipeline) throws Exception {
        if (keyPointStageName == null) {
            return null;
        }
        Result result = pipeline.getResult(keyPointStageName);
        if (result == null || result.model == null) {
            return null;
        }
        Object model = result.model;
        if (model instanceof List) {
            List<Circle> points = new ArrayList<>();
            for (Object o : ((List) model)) {
                points.add(convertKeyPointToCircle(o));
            }
            return new Result(null, points);
        }
        else {
            return new Result(null, convertKeyPointToCircle(model));
        }
    }

    private static Circle convertKeyPointToCircle(Object keyPointHolder) throws Exception {
        if (keyPointHolder instanceof KeyPoint) {
        	KeyPoint kp = (KeyPoint) keyPointHolder;
            return new Circle(kp.pt.x, kp.pt.y, 2.0 * (double)kp.size);
        }
        else if (keyPointHolder instanceof Point) {
            Point point = (Point) keyPointHolder;
            return new Circle(point.x, point.y, 0);
        }
        else {
            throw new Exception("Don't know how to convert " + keyPointHolder + "to KeyPoint.");
        }
    }
}
