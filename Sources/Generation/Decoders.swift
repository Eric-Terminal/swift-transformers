#if canImport(CoreML)
import CoreML

// MARK: Greedy Decoding

@available(macOS 15.0, iOS 18.0, watchOS 11.0, *)
func selectNextTokenUsingGreedyDecoding(from scores: MLTensor) -> MLTensor {
    if #available(watchOS 11.0, *) {
        return scores.argmax(alongAxis: -1).reshaped(to: [1, 1])
    } else {
        // Fallback on earlier versions
        fatalError("MLTensor is not available on this OS version")
    }
}

// MARK: Top-K Sampling

@available(macOS 15.0, iOS 18.0, watchOS 11.0, *)
func selectNextTokenUsingTopKSampling(from scores: MLTensor, temperature: Float, topK: Int) -> MLTensor {
    if #available(watchOS 11.0, *) {
        let temperatureAdjustedScores = scores / temperature
        let (topKScores, topKIndices) = temperatureAdjustedScores.topK(topK)
        let topKProbs = topKScores.softmax(alongAxis: -1)
        let rnd = topKProbs.sum() * Float.random(in: 0..<1)
        var accumTopKProbs = topKProbs.cumulativeSum(alongAxis: -1)
        accumTopKProbs += (accumTopKProbs .< rnd) * 100.0
        let topKIndex = accumTopKProbs.argsort()[..., 0]
        let nextTokenTensor = topKIndices.gathering(
            atIndices: topKIndex,
            alongAxis: topKIndices.rank - 1
        )
        return nextTokenTensor.reshaped(to: [1, 1])
    } else {
        // Fallback on earlier versions
        fatalError("MLTensor is not available on this OS version")
    }
}
#endif // canImport(CoreML)
