import { Progress } from "@kobalte/core/progress";
import { useNetworkStore } from "../../stores/networkStore";
import { Index, Match, Show, Switch } from "solid-js";

export default function Status() {
    const { progress, hyperparameter, prediction, trainAcc, testAcc } = useNetworkStore();

    return (
        <div class="p-3">
            <Switch fallback={
                <div class="flex justify-center items-center">
                    Start by training a network
                </div>
            }>
                <Match when={!prediction() && progress() !== null ? progress() : null}>
                    {(currentProgress) => <>
                        <Progress
                            value={currentProgress()}
                            maxValue={hyperparameter.targetEpochs()}
                            getValueLabel={() => `Epoch ${currentProgress()} / ${hyperparameter.targetEpochs()}`}
                            class="flex flex-col items-between w-full"
                        >
                            <div class="flex justify-between">
                                <Progress.Label>Training...</Progress.Label>
                                <Progress.ValueLabel />
                            </div>
                            <Progress.Track class="bg-surface w-full h-[16px] relative rounded-xs">
                                <Progress.Fill class="w-(--kb-progress-fill-width) h-full bg-accent rounded-xs transition-[width] duration-250 ease-linear" />
                            </Progress.Track>
                        </Progress>
                        <Show when={progress() == hyperparameter.targetEpochs() && trainAcc() && testAcc()}>
                            Training done. Draw a number
                        </Show>
                    </>
                    }
                </Match>
                <Match when={prediction() ? prediction() : null}>
                    {(pred) => <div class="grid grid-rows-5 gap-x-2 grid-flow-col">

                        <Index each={pred()}>
                            {(v, i) =>
                                <Progress
                                    value={v()}
                                    maxValue={1}
                                    getValueLabel={() => v().toPrecision(4)}
                                    class="flex w-full items-center h-full gap-2"
                                >
                                    <Progress.Label class="text-sm">{i}</Progress.Label>
                                    <Progress.Track class="w-full h-[8px] relative rounded-xs">
                                        <Progress.Fill
                                            class="w-(--kb-progress-fill-width) h-full bg-accent
                                            rounded-xs transition-[width] duration-250 ease-linear" >
                                        </Progress.Fill>
                                    </Progress.Track>
                                </Progress>
                            }
                        </Index>
                    </div>
                    }
                </Match>
            </Switch>
        </div>
    );
}
