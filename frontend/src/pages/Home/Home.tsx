import MnistCanvas from "./MnistCanvas";
import Settings from "./Settings";
import Status from "./Status";

export default function Home() {
    return <div class="pt-3 sm:pt-0 h-dvh w-dvw grid grid-cols-1 grid-rows-[auto_1fr_auto] sm:grid-cols-[30%_70%] sm:grid-rows-[1fr_auto]">
        <Settings />
        <MnistCanvas />
        <Status />
    </div>
}
