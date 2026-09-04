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
        ctx.lineWidth = 16; // thick line, similar stroke weight to MNIST digits
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

    function downscale() {
        const smallCtx = smallCanvas.getContext("2d")!;
        smallCtx.imageSmoothingEnabled = true;
        smallCtx.drawImage(
            bigCanvas,
            0, 0, bigCanvas.width, bigCanvas.height,
            0, 0, SMALL_SIZE, SMALL_SIZE
        );
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
