import { ComponentProps, Index, Show } from "solid-js"
import { useNetworkStore } from "../../stores/networkStore"
import { NumberField } from "@kobalte/core/number-field"
import { Button } from "@kobalte/core/button"
import { Plus, Trash2 } from "lucide-solid"
import { Slider } from "@kobalte/core/slider"

type LayerFieldProps = {
    index: number;
    value: number;
    editable: boolean;
    onChange: (value: number) => void;
    onRemove: () => void;
};

export default function Settings() {
    const { hyperparameter, train, createNetwork, isTraining } = useNetworkStore();
    return <div class="flex sm:flex-col p-3 h-full min-h-0 gap-8 dark:bg-neutral-900 bg-neutral-300
    border-t-accent sm:border-r-accent border-t-2 sm:border-r-2 sm:rounded-r-xl row-span-2 order-1 sm:order-none
    rounded-t-xl sm:border-t-0 sm:rounded-t-none">
        <Architecture class="overflow-y-auto h-full min-h-0 min-w-0 max-h-[300px] sm:max-h-none grow shrink basis-auto sm:flex-1 pr-3 [scrollbar-gutter:stable]" />
        <div class="sm:shrink-0 min-w-0 flex flex-col gap-3">
            <Slider minValue={-3} maxValue={1} step={0.01} defaultValue={[-2]} onChange={([v]) => {
                hyperparameter.setLr(Math.pow(10, v));
            }}
                getValueLabel={_ => hyperparameter.lr().toPrecision(4)}
                class="flex flex-col items-between w-full relative">
                <div class="flex justify-between">
                    <Slider.Label class="text-sm">Learning Rate</Slider.Label>
                    <Slider.ValueLabel />
                </div>
                <Slider.Track class="bg-surface w-full h-[8px] relative rounded-full">
                    <Slider.Fill class="bg-secondary h-full absolute rounded-full" />
                    <Slider.Thumb class="bg-white w-[16px] h-[16px] block rounded-full top-[-4px] border-1 border-secondary">
                        <Slider.Input />
                    </Slider.Thumb>
                </Slider.Track>
            </Slider>

            <div class="flex gap-5">
                <NumberField
                    value={hyperparameter.batchSize()}
                    minValue={1}
                    maxValue={60000}
                    onRawValueChange={hyperparameter.setBatchSize}
                    class="flex flex-col gap-0 min-w-17 flex-grow-1"
                >
                    <NumberField.Label class="flex items-center text-sm">Batch size</NumberField.Label>
                    <NumberField.Input class="bg-surface p-1 rounded-sm focus-visible:outline-1 focus-visible:outline-secondary" />
                </NumberField>

                <NumberField
                    value={hyperparameter.epochs()}
                    minValue={0}
                    onRawValueChange={hyperparameter.setEpochs}
                    class="flex flex-col gap-0 min-w-15 flex-grow-1"
                >
                    <NumberField.Label class="flex items-center text-sm">Epochs</NumberField.Label>
                    <NumberField.Input class="bg-surface p-1 rounded-sm focus-visible:outline-1 focus-visible:outline-secondary" />
                </NumberField>
            </div>
            <Button class="bg-accent mt-4 w-full rounded-sm p-1 disabled:cursor-not-allowed data-[disabled]:opacity-50
            data-[disabled]:cursor-not-allowed data-[disabled]:pointer-events-none"
                disabled={isTraining()} onclick={
                    async () => {
                        hyperparameter.setTargetEpochs(hyperparameter.epochs());
                        await createNetwork()
                        const [trainAcc, testAcc] = await train()
                        console.log(trainAcc, testAcc);
                    }
                }>Train network</Button>
        </div>
    </div >
}

function Architecture(props: ComponentProps<"div">) {
    const { hyperparameter } = useNetworkStore();
    const arch = () => hyperparameter.architecture();

    function addLayer() {
        const current = arch();
        const insertAt = current.length - 1;
        const newLayer = 32;
        hyperparameter.setArchitecture(current.toSpliced(insertAt, 0, newLayer));
    }

    return (
        <div {...props}>
            <div class="flex flex-col gap-2">
                <Index each={arch()}>
                    {(el, i) => (
                        <LayerField
                            index={i}
                            value={el()}
                            editable={i !== 0 && i !== arch().length - 1}
                            onChange={(v) => {
                                if (Number.isNaN(v)) return;
                                const newArch = arch().map((old, idx) => (idx === i ? v : old));
                                hyperparameter.setArchitecture(newArch);
                            }}
                            onRemove={() => {
                                hyperparameter.setArchitecture(arch().toSpliced(i, 1));
                            }}
                        />
                    )}
                </Index>
            </div >
            <Button onClick={addLayer} class="flex items-center gap-1 bg-primary rounded-sm text-black p-1 mt-3">
                <Plus size={16} /> Add Layer
            </Button>
        </div>
    );
}

function LayerField(props: LayerFieldProps) {
    return (
        <div class="flex  gap-3">
            <NumberField
                value={props.value}
                minValue={1}
                disabled={!props.editable}
                onRawValueChange={props.onChange}
                class="flex flex-col gap-0 w-full"
            >
                <NumberField.Label class="flex items-center text-sm data-[disabled]:opacity-50">Layer {props.index}</NumberField.Label>
                <NumberField.Input class="bg-surface p-1 w-full rounded-sm focus-visible:outline-1 focus-visible:outline-secondary  data-[disabled]:cursor-not-allowed data-[disabled]:bg-surface/50" />
            </NumberField>

            <div class="w-9 shrink-0 flex flex-col gap-0">
                <div class="text-sm invisible">&nbsp;</div>
                <Show when={props.editable}>
                    <Button onClick={props.onRemove} aria-label="Remove layer" class="bg-secondary p-2 rounded-sm">
                        <Trash2 size={16} />
                    </Button>
                </Show>
            </div>
        </div>
    );
}
