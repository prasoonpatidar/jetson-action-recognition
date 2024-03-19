import Foundation
import AVFoundation
import Vision
import Combine

struct requestPayload:Codable {
    let bodyParts_x :[Double]
    let bodyParts_y :[Double]
    let bodyParts_c :[Float]
    let pose_ts: Int64
    init(ts:Int64, bodyParts: [VNHumanBodyPoseObservation.JointName : VNRecognizedPoint]){
        self.pose_ts = ts
        self.bodyParts_x = [
            bodyParts[.nose]!.location.x,
            bodyParts[.leftEye]!.location.x,
            bodyParts[.rightEye]!.location.x,
            bodyParts[.leftEar]!.location.x,
            bodyParts[.rightEar]!.location.x,
            bodyParts[.leftShoulder]!.location.x,
            bodyParts[.rightShoulder]!.location.x,
            bodyParts[.leftElbow]!.location.x,
            bodyParts[.rightElbow]!.location.x,
            bodyParts[.leftWrist]!.location.x,
            bodyParts[.rightWrist]!.location.x,
            bodyParts[.leftHip]!.location.x,
            bodyParts[.rightHip]!.location.x,
            bodyParts[.leftKnee]!.location.x,
            bodyParts[.rightKnee]!.location.x,
            bodyParts[.leftAnkle]!.location.x,
            bodyParts[.rightAnkle]!.location.x,
        ]
        self.bodyParts_y = [
            bodyParts[.nose]!.location.y,
            bodyParts[.leftEye]!.location.y,
            bodyParts[.rightEye]!.location.y,
            bodyParts[.leftEar]!.location.y,
            bodyParts[.rightEar]!.location.y,
            bodyParts[.leftShoulder]!.location.y,
            bodyParts[.rightShoulder]!.location.y,
            bodyParts[.leftElbow]!.location.y,
            bodyParts[.rightElbow]!.location.y,
            bodyParts[.leftWrist]!.location.y,
            bodyParts[.rightWrist]!.location.y,
            bodyParts[.leftHip]!.location.y,
            bodyParts[.rightHip]!.location.y,
            bodyParts[.leftKnee]!.location.y,
            bodyParts[.rightKnee]!.location.y,
            bodyParts[.leftAnkle]!.location.y,
            bodyParts[.rightAnkle]!.location.y,

        ]
        self.bodyParts_c = [
            bodyParts[.nose]!.confidence,
            bodyParts[.leftEye]!.confidence,
            bodyParts[.rightEye]!.confidence,
            bodyParts[.leftEar]!.confidence,
            bodyParts[.rightEar]!.confidence,
            bodyParts[.leftShoulder]!.confidence,
            bodyParts[.rightShoulder]!.confidence,
            bodyParts[.leftElbow]!.confidence,
            bodyParts[.rightElbow]!.confidence,
            bodyParts[.leftWrist]!.confidence,
            bodyParts[.rightWrist]!.confidence,
            bodyParts[.leftHip]!.confidence,
            bodyParts[.rightHip]!.confidence,
            bodyParts[.leftKnee]!.confidence,
            bodyParts[.rightKnee]!.confidence,
            bodyParts[.leftAnkle]!.confidence,
            bodyParts[.rightAnkle]!.confidence,

        ]
    }
}

extension Date {
    func currentTimeNanosecs() -> Int64 {
        return Int64(self.timeIntervalSince1970 * 1e9)
    }
    func currentTimeMillisecs() -> Int64{
        return Int64(self.timeIntervalSince1970 * 1000)
    }
}

class PoseEstimator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, ObservableObject {
    let sequenceHandler = VNSequenceRequestHandler()    
    @Published var bodyParts = [VNHumanBodyPoseObservation.JointName : VNRecognizedPoint]()
    var current_time = Date().currentTimeMillisecs()
    @Published var frames_per_sec = 0
    var numFrames:Double = 0
    
    let send_url = "http://edusense-compute-4.andrew.cmu.edu:9090/pose_info_iphone"
    
    var subscriptions = Set<AnyCancellable>()
    
    override init() {
        super.init()
        $bodyParts
            .dropFirst()
            .sink(receiveValue: { bodyParts in self.countSquats(bodyParts: bodyParts)})
            .store(in: &subscriptions)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let humanBodyRequest = VNDetectHumanBodyPoseRequest(completionHandler: detectedBodyPose)
        do {
            try sequenceHandler.perform(
              [humanBodyRequest],
              on: sampleBuffer,
                orientation: .right)
        } catch {
          print(error.localizedDescription)
        }
    }
    func detectedBodyPose(request: VNRequest, error: Error?) {
        guard let bodyPoseResults = request.results as? [VNHumanBodyPoseObservation]
          else { return }
        guard let bodyParts = try? bodyPoseResults.first?.recognizedPoints(.all) else { return }
        DispatchQueue.main.async {
            self.bodyParts = bodyParts
            self.sendPoseData(bodyParts: bodyParts)
            self.numFrames+=1
            let curr_time=Date().currentTimeMillisecs()
            if (curr_time-self.current_time) > Int64(1e3){
                self.frames_per_sec = Int(self.numFrames)
                self.current_time = Date().currentTimeMillisecs()
                self.numFrames = 0
            }
        }
    }
    
    func sendPoseData(bodyParts: [VNHumanBodyPoseObservation.JointName : VNRecognizedPoint]) {
        let curr_timestamp = Date().currentTimeNanosecs()
        let requestPoseData = requestPayload(ts: curr_timestamp, bodyParts: bodyParts)
//        let encodedPoseData = try? JSONSerialization.data(withJSONObject: requestPoseData,options: [])
        
        let encodedPoseData = try? JSONEncoder().encode(requestPoseData)
//        print(String(data: encodedPoseData!, encoding: .utf8))
    
        let url = URL(string: self.send_url)!
        var request = URLRequest(url: url)
        request.setValue(
            "application/json",
            forHTTPHeaderField: "Content-Type"
        )
        
        request.httpMethod = "POST"
        if let encodedPoseData{
            request.httpBody = encodedPoseData
        } else {
            request.httpBody = try? JSONSerialization.data(withJSONObject: ["Empty"])
            print("Empty JSON Object")
        }
        let session = URLSession.shared
        let task = session.dataTask(with: request) { (data, response, error) in

            if let data = data, let dataString = String(data: data, encoding: .utf8) {
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode==200{
                        
                    }else{
                        print(dataString)
                        print(httpResponse.statusCode)
                    }
                }

            }

        }
        task.resume()
    }
    
    func countSquats(bodyParts: [VNHumanBodyPoseObservation.JointName : VNRecognizedPoint]) {
        
//        let rightKnee = bodyParts[.rightKnee]!.location
//        let leftKnee = bodyParts[.rightKnee]!.location
//        let rightHip = bodyParts[.rightHip]!.location
//        let rightAnkle = bodyParts[.rightAnkle]!.location
//        let leftAnkle = bodyParts[.leftAnkle]!.location
//
//        let firstAngle = atan2(rightHip.y - rightKnee.y, rightHip.x - rightKnee.x)
//        let secondAngle = atan2(rightAnkle.y - rightKnee.y, rightAnkle.x - rightKnee.x)
//        var angleDiffRadians = firstAngle - secondAngle
//        while angleDiffRadians < 0 {
//                    angleDiffRadians += CGFloat(2 * Double.pi)
//                }
//        let angleDiffDegrees = Int(angleDiffRadians * 180 / .pi)
//        if angleDiffDegrees > 150 && self.wasInBottomPosition {
//            self.squatCount += 1
//            self.wasInBottomPosition = false
//        }
//
//        let hipHeight = rightHip.y
//        let kneeHeight = rightKnee.y
//        if hipHeight < kneeHeight {
//            self.wasInBottomPosition = true
//        }
//
//
//        let kneeDistance = rightKnee.distance(to: leftKnee)
//        let ankleDistance = rightAnkle.distance(to: leftAnkle)
//
//        if ankleDistance > kneeDistance {
//            self.isGoodPosture = false
//        } else {
//            self.isGoodPosture = true
//        }
//
    }

}
