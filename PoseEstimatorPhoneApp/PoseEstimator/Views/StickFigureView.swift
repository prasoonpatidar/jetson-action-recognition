import SwiftUI

struct StickFigureView: View {
    @ObservedObject var poseEstimator: PoseEstimator
    var size: CGSize
    var body: some View {
        if poseEstimator.bodyParts.isEmpty == false {
            ZStack {
                // Right leg
                if poseEstimator.bodyParts[.rightAnkle]!.confidence>0, poseEstimator.bodyParts[.rightKnee]!.confidence>0, poseEstimator.bodyParts[.rightHip]!.confidence>0,poseEstimator.bodyParts[.root]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.rightAnkle]!.location, poseEstimator.bodyParts[.rightKnee]!.location, poseEstimator.bodyParts[.rightHip]!.location,
                                   poseEstimator.bodyParts[.root]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.rightKnee]!.confidence>0, poseEstimator.bodyParts[.rightHip]!.confidence>0,poseEstimator.bodyParts[.root]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.rightKnee]!.location, poseEstimator.bodyParts[.rightHip]!.location,
                                   poseEstimator.bodyParts[.root]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.rightHip]!.confidence>0,poseEstimator.bodyParts[.root]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.rightHip]!.location,
                                   poseEstimator.bodyParts[.root]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                // Left leg
                if poseEstimator.bodyParts[.leftAnkle]!.confidence>0, poseEstimator.bodyParts[.leftKnee]!.confidence>0, poseEstimator.bodyParts[.leftHip]!.confidence>0,poseEstimator.bodyParts[.root]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.leftAnkle]!.location, poseEstimator.bodyParts[.leftKnee]!.location, poseEstimator.bodyParts[.leftHip]!.location,
                                   poseEstimator.bodyParts[.root]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.leftKnee]!.confidence>0, poseEstimator.bodyParts[.leftHip]!.confidence>0,poseEstimator.bodyParts[.root]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.leftKnee]!.location, poseEstimator.bodyParts[.leftHip]!.location,
                                   poseEstimator.bodyParts[.root]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.leftHip]!.confidence>0,poseEstimator.bodyParts[.root]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.leftHip]!.location,
                                   poseEstimator.bodyParts[.root]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                
                // Right arm
                if poseEstimator.bodyParts[.rightWrist]!.confidence>0, poseEstimator.bodyParts[.rightElbow]!.confidence>0, poseEstimator.bodyParts[.rightShoulder]!.confidence>0,poseEstimator.bodyParts[.neck]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.rightWrist]!.location, poseEstimator.bodyParts[.rightElbow]!.location, poseEstimator.bodyParts[.rightShoulder]!.location, poseEstimator.bodyParts[.neck]!.location], size: size)
                        .stroke(lineWidth: 5.0)
                        .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.rightElbow]!.confidence>0, poseEstimator.bodyParts[.rightShoulder]!.confidence>0,poseEstimator.bodyParts[.neck]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.rightElbow]!.location, poseEstimator.bodyParts[.rightShoulder]!.location, poseEstimator.bodyParts[.neck]!.location], size: size)
                        .stroke(lineWidth: 5.0)
                        .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.rightShoulder]!.confidence>0,poseEstimator.bodyParts[.neck]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.rightShoulder]!.location, poseEstimator.bodyParts[.neck]!.location], size: size)
                        .stroke(lineWidth: 5.0)
                        .fill(Color.blue)
                }
                
                // Left arm
                if poseEstimator.bodyParts[.leftWrist]!.confidence>0, poseEstimator.bodyParts[.leftElbow]!.confidence>0, poseEstimator.bodyParts[.leftShoulder]!.confidence>0,poseEstimator.bodyParts[.neck]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.leftWrist]!.location, poseEstimator.bodyParts[.leftElbow]!.location, poseEstimator.bodyParts[.leftShoulder]!.location, poseEstimator.bodyParts[.neck]!.location], size: size)
                        .stroke(lineWidth: 5.0)
                        .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.leftElbow]!.confidence>0, poseEstimator.bodyParts[.leftShoulder]!.confidence>0,poseEstimator.bodyParts[.neck]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.leftElbow]!.location, poseEstimator.bodyParts[.leftShoulder]!.location, poseEstimator.bodyParts[.neck]!.location], size: size)
                        .stroke(lineWidth: 5.0)
                        .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.leftShoulder]!.confidence>0,poseEstimator.bodyParts[.neck]!.confidence>0{
                    Stick(points: [poseEstimator.bodyParts[.leftShoulder]!.location, poseEstimator.bodyParts[.neck]!.location], size: size)
                        .stroke(lineWidth: 5.0)
                        .fill(Color.blue)
                }
                
                
                // Root to nose
                if poseEstimator.bodyParts[.root]!.confidence>0, poseEstimator.bodyParts[.neck]!.confidence>0, poseEstimator.bodyParts[.nose]!.confidence>0 {
                    Stick(points: [poseEstimator.bodyParts[.root]!.location,
                                   poseEstimator.bodyParts[.neck]!.location,  poseEstimator.bodyParts[.nose]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }
                else if poseEstimator.bodyParts[.neck]!.confidence>0, poseEstimator.bodyParts[.nose]!.confidence>0 {
                    Stick(points: [poseEstimator.bodyParts[.neck]!.location,  poseEstimator.bodyParts[.nose]!.location], size: size)
                    .stroke(lineWidth: 5.0)
                    .fill(Color.blue)
                }

                }
            }
        }
}

