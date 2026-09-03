import MnistCanvas from "./MnistCanvas";
import Settings from "./Settings";
import Status from "./Status";

export default function Home() {
    return <div class="h-dvh w-dvw grid grid-cols-[30%_70%] grid-rows-[1fr_auto]">
        <Settings />
        <MnistCanvas />
        <Status />
    </div>
}
