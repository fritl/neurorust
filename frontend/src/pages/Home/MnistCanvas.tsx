import { Button } from "@kobalte/core/button";
import { onMount } from "solid-js";
import { useNetworkStore } from "../../stores/networkStore";

export default function MnistCanvas() {
    const { predict, network, isTraining } = useNetworkStore();
    let bigCanvas!: HTMLCanvasElement;
    let smallCanvas!: HTMLCanvasElement;

    const BIG_SIZE = 280; // CSS pixels, 10x scale of 28
    const SMALL_SIZE = 28;

    let isDrawing = false;

    function setupCanvas(canvas: HTMLCanvasElement, cssSize: number) {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = cssSize * dpr;
        canvas.height = cssSize * dpr;
        canvas.style.width = `${cssSize}px`;
        canvas.style.height = `${cssSize}px`;

        const ctx = canvas.getContext("2d")!;
        ctx.scale(dpr, dpr); // 1 unit in draw calls = 1 CSS pixel
        return ctx;
    } let ctx: CanvasRenderingContext2D;

    onMount(() => {
        ctx = setupCanvas(bigCanvas, BIG_SIZE);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = 30; // thick line, similar stroke weight to MNIST digits
        ctx.strokeStyle = "white";
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, BIG_SIZE, BIG_SIZE); // black background, MNIST style
    });

    function getPos(e: PointerEvent) {
        const rect = bigCanvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    }

    function startDraw(e: PointerEvent) {
        isDrawing = true;
        const { x, y } = getPos(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function draw(e: PointerEvent) {
        if (!isDrawing) return;
        const { x, y } = getPos(e);
        ctx.lineTo(x, y);
        ctx.stroke();
    }

    function stopDraw() {
        isDrawing = false;
        downscale();
        const x = getImageDataAsFloat32();
        if (!network() || isTraining()) return;
        predict(x);
    }

    // MNIST-style normalization: the digit's bounding box is scaled
    // (uniformly, so it never gets distorted) to fit inside a 20x20 area
    // and then centered within the 28x28 preview, matching the
    // preprocessing MNIST itself uses.
    const NORMALIZED_SIZE = 20;

    function getBoundingBox(imgData: ImageData): { minX: number; minY: number; maxX: number; maxY: number } | null {
        const { data, width, height } = imgData;
        let minX = width, minY = height, maxX = -1, maxY = -1;
        const threshold = 20; // ignore near-black background noise

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const value = data[(y * width + x) * 4]; // red channel, image is grayscale
                if (value > threshold) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX < 0 || maxY < 0) return null; // nothing drawn
        return { minX, minY, maxX, maxY };
    }

    function downscaleCanvas(source: HTMLCanvasElement, sw: number, sh: number, dw: number, dh: number): HTMLCanvasElement {
        let curCanvas = source;
        let curW = sw;
        let curH = sh;

        // Halve the size repeatedly until close to the target.
        // Each step is a small ratio, so the browser's built-in
        // interpolation actually averages pixels instead of skipping them.
        while (curW / 2 > dw && curH / 2 > dh) {
            const nextW = Math.floor(curW / 2);
            const nextH = Math.floor(curH / 2);

            const stepCanvas = document.createElement("canvas");
            stepCanvas.width = nextW;
            stepCanvas.height = nextH;

            const stepCtx = stepCanvas.getContext("2d")!;
            stepCtx.imageSmoothingEnabled = true;
            stepCtx.imageSmoothingQuality = "high";
            stepCtx.drawImage(curCanvas, 0, 0, curW, curH, 0, 0, nextW, nextH);

            curCanvas = stepCanvas;
            curW = nextW;
            curH = nextH;
        }

        const finalCanvas = document.createElement("canvas");
        finalCanvas.width = dw;
        finalCanvas.height = dh;
        const finalCtx = finalCanvas.getContext("2d")!;
        finalCtx.imageSmoothingEnabled = true;
        finalCtx.imageSmoothingQuality = "high";
        finalCtx.drawImage(curCanvas, 0, 0, curW, curH, 0, 0, dw, dh);
        return finalCanvas;
    }

    function downscale() {
        const smallCtx = smallCanvas.getContext("2d")!;
        smallCtx.imageSmoothingEnabled = true;
        smallCtx.imageSmoothingQuality = "high";

        const fullCtx = bigCanvas.getContext("2d")!;
        const bbox = getBoundingBox(fullCtx.getImageData(0, 0, bigCanvas.width, bigCanvas.height));

        smallCtx.fillStyle = "black";
        smallCtx.fillRect(0, 0, SMALL_SIZE, SMALL_SIZE);

        if (!bbox) return; // nothing drawn, leave preview blank

        const { minX, minY, maxX, maxY } = bbox;
        const bboxW = maxX - minX + 1;
        const bboxH = maxY - minY + 1;

        // Crop the digit out of the big canvas.
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = bboxW;
        cropCanvas.height = bboxH;
        cropCanvas.getContext("2d")!.drawImage(bigCanvas, minX, minY, bboxW, bboxH, 0, 0, bboxW, bboxH);

        // Scale uniformly (same factor on both axes) so the longer side
        // becomes NORMALIZED_SIZE, keeping the digit's proportions intact.
        const scale = NORMALIZED_SIZE / Math.max(bboxW, bboxH);
        const targetW = Math.max(1, Math.round(bboxW * scale));
        const targetH = Math.max(1, Math.round(bboxH * scale));

        const resized = downscaleCanvas(cropCanvas, bboxW, bboxH, targetW, targetH);

        // Center the normalized digit inside the SMALL_SIZE preview.
        const destX = Math.round((SMALL_SIZE - targetW) / 2);
        const destY = Math.round((SMALL_SIZE - targetH) / 2);
        smallCtx.drawImage(resized, destX, destY);
    }

    function getImageDataAsFloat32(): Float32Array {
        const smallCtx = smallCanvas.getContext("2d")!;
        const imgData = smallCtx.getImageData(0, 0, SMALL_SIZE, SMALL_SIZE);
        const pixels = imgData.data;

        const float32 = new Float32Array(SMALL_SIZE * SMALL_SIZE); // 784 Längen-Array

        for (let i = 0; i < float32.length; i++) {
            float32[i] = pixels[i * 4] / 255.0;
        }

        return float32;
    }

    function clear() {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, BIG_SIZE, BIG_SIZE);
        const smallCtx = smallCanvas.getContext("2d")!;
        smallCtx.clearRect(0, 0, SMALL_SIZE, SMALL_SIZE);
    }

    return (
        <div class="flex flex-col justify-center gap-5 place-self-center">
            <div class="flex gap-3 place-self-center flex-wrap justify-center">
                <canvas
                    ref={bigCanvas}
                    onPointerDown={startDraw}
                    onPointerMove={draw}
                    onPointerUp={stopDraw}
                    onPointerLeave={stopDraw}
                    class="bg-black rounded-sm touch-none border-surface border-1"

                />
                <canvas
                    ref={smallCanvas}
                    width={SMALL_SIZE}
                    height={SMALL_SIZE}
                    style={{ width: "112px", height: "112px", "image-rendering": "pixelated" }}
                    class="bg-black rounded-sm border-1 border-surface"
                />
            </div>
            <Button onClick={clear} class="bg-secondary p-2 ml-4 mr-4 sm:ml-0 sm:mr-0 rounded-sm">Clear</Button>
        </div>
    );
}
