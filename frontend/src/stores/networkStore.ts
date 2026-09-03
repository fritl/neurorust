import { Accessor, createSignal, Setter } from "solid-js";
import init, { WasmNetwork } from "wasm-backend";

type MnistData = {
    trainImages: Uint8Array,
    trainLabels: Uint8Array,
    testImages: Uint8Array,
    testLabels: Uint8Array
}

type HyperParameterType = {
    architecture: Accessor<number[]>,
    setArchitecture: Setter<number[]>,
    lr: Accessor<number>,
    setLr: Setter<number>,
    batchSize: Accessor<number>,
    setBatchSize: Setter<number>,
    epochs: Accessor<number>,
    setEpochs: Setter<number>
    targetEpochs: Accessor<number>
    setTargetEpochs: Setter<number>
    seed: Accessor<bigint | null>,
    setSeed: Setter<bigint | null>
}

type NetworkStoreType = {
    createNetwork: () => Promise<void>,
    train: () => Promise<[number, number]>,
    predict: (x: Float32Array) => Promise<void>,
    progress: () => number | null,
    stop_training: () => void,
    prediction: Accessor<number[] | null>,
    testAcc: Accessor<number | null>,
    trainAcc: Accessor<number | null>,
    hyperparameter: HyperParameterType,
    isTraining: Accessor<boolean>,
    network: Accessor<WasmNetwork | null>
}


// async function fetchAndDecompress(url: string): Promise<Uint8Array> {
//     const response = await fetch(url);
//     if (!response.body) throw new Error(`Download from ${url} failed.`);
//     const decompressedStream = response.body.pipeThrough(new DecompressionStream('gzip'));
//     console.log("Hello");
//     const buffer = await new Response(decompressedStream).arrayBuffer();
//     console.log("HI");
//     return new Uint8Array(buffer);
// }

async function fetchAndDecompress(url: string): Promise<Uint8Array> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download from ${url} failed: ${response.status}`);
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

async function fetchMnistData(): Promise<MnistData> {
    const [train_images, train_labels, test_images, test_labels] = await Promise.all([
        fetchAndDecompress("/data/train-images-idx3-ubyte.gz"),
        fetchAndDecompress("/data/train-labels-idx1-ubyte.gz"),
        fetchAndDecompress("/data/t10k-images-idx3-ubyte.gz"),
        fetchAndDecompress("/data/t10k-labels-idx1-ubyte.gz"),
    ]);
    return { trainImages: train_images, trainLabels: train_labels, testImages: test_images, testLabels: test_labels };
}

const [architecture, setArchitecture] = createSignal([784, 128, 10]);
const [lr, setLr] = createSignal(0.01);
const [batchSize, setBatchSize] = createSignal(128);
const [epochs, setEpochs] = createSignal(100);
const [targetEpochs, setTargetEpochs] = createSignal(epochs());
const [seed, setSeed] = createSignal<bigint | null>(null);

const hyperparameter: HyperParameterType = { architecture, setArchitecture, lr, setLr, batchSize, setBatchSize, epochs, setEpochs, seed, setSeed, targetEpochs, setTargetEpochs }

const [mnistData] = createSignal<Promise<MnistData>>(fetchMnistData());
const [network, setNetwork] = createSignal<WasmNetwork | null>(null);
const [progress, setProgress] = createSignal<number | null>(null);
const [prediction, setPrediction] = createSignal<number[] | null>(null);
const [trainAcc, setTrainAcc] = createSignal<number | null>(null);
const [testAcc, setTestAcc] = createSignal<number | null>(null);
const [isTraining, setIsTraining] = createSignal(false);
const wasmInitPromise: Promise<void> = init().then(() => { });

export function useNetworkStore(): NetworkStoreType {
    const createNetwork = async () => {
        await wasmInitPromise;
        const data = await mnistData();
        setNetwork(null);
        setProgress(null);
        setPrediction(null);
        setTrainAcc(null);
        setTestAcc(null);

        const net = await WasmNetwork.network(
            new Uint32Array(hyperparameter.architecture()),
            hyperparameter.lr(),
            hyperparameter.seed(),
            data.trainImages,
            data.trainLabels,
            data.testImages,
            data.testLabels
        );
        setNetwork(net);
    };

    const train = async (): Promise<[number, number]> => {
        setIsTraining(true)
        const net = network();
        if (!net) {
            setIsTraining(false);
            throw new Error("network not created");
        }
        setProgress(null);
        setPrediction(null);
        setTrainAcc(null);
        setTestAcc(null);
        const result = await net.train(hyperparameter.epochs(), hyperparameter.batchSize(), setProgress)
        if (result.length !== 2) throw new Error(`Expected a pair, got ${result.length} values`)
        setTrainAcc(result[0])
        setTestAcc(result[1])
        setIsTraining(false);
        return [result[0], result[1]];
    }

    const predict = async (x: Float32Array) => {
        const net = network();
        if (!net) throw new Error("network not created");
        const pred = await net.predict(x);
        setPrediction(Array.from(pred));
    }

    const stop_training = () => {
        const net = network();
        if (!net) throw new Error("network not created");
        return net.abort_training();
    }


    return { createNetwork, train, predict, progress, stop_training, hyperparameter, prediction, trainAcc, testAcc, isTraining, network };
}
