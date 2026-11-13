import { createSignal } from "solid-js";
import MnistCanvas from "./MnistCanvas";
import Settings from "./Settings";
import Status from "./Status";

export default function Home() {
    const [collapsed, setCollapsed] = createSignal(false);
    return <div class={`pt-3 sm:pt-0 sm:h-dvh w-dvw grid grid-cols-1 grid-rows-[auto_1fr_auto] sm:grid-rows-[1fr_auto] transition-[grid-template-columns] duration-200 ${collapsed() ? "sm:grid-cols-[auto_1fr]" : "sm:grid-cols-[30%_70%]"}`}>
        <Settings collapsed={collapsed()} onToggle={() => setCollapsed(c => !c)} />
        <MnistCanvas />
        <Status />
    </div>
}
