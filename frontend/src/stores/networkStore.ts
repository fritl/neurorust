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
    seed: Accessor<bigint | null>,
    setSeed: Setter<bigint | null>
}

type NetworkStoreType = {
    createNetwork: () => Promise<void>,
    train: () => Promise<[number, number]>,
    predict: (x: Float32Array) => Promise<Float32Array>,
    progress: () => number | null,
    stop_training: () => void,
    hyperparameter: HyperParameterType
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
const [seed, setSeed] = createSignal<bigint | null>(null);

const hyperparameter: HyperParameterType = { architecture, setArchitecture, lr, setLr, batchSize, setBatchSize, epochs, setEpochs, seed, setSeed }

const [mnistData] = createSignal<Promise<MnistData>>(fetchMnistData());
const [network, setNetwork] = createSignal<WasmNetwork | null>(null);
const [progress, setProgress] = createSignal<number | null>(null);
const wasmInitPromise: Promise<void> = init().then(() => { });

export function useNetworkStore(): NetworkStoreType {
    const createNetwork = async () => {
        await wasmInitPromise;
        const data = await mnistData();
        setProgress(null);

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
        const net = network();
        if (!net) throw new Error("network not created");
        const result = await net.train(hyperparameter.epochs(), hyperparameter.epochs(), setProgress)
        if (result.length !== 2) throw new Error(`Expected a pair, got ${result.length} values`)
        return [result[0], result[1]];
    }

    const predict = async (x: Float32Array) => {
        const net = network();
        if (!net) throw new Error("network not created");
        return await net.predict(x);
    }

    const stop_training = () => {
        const net = network();
        if (!net) throw new Error("network not created");
        return net.abort_training();
    }


    return { createNetwork, train, predict, progress, stop_training, hyperparameter };
}
