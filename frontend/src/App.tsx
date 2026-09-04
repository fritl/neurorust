import { Show, type Component } from 'solid-js';
import Home from './pages/Home/Home';
import { isLoading } from './stores/networkStore';

const App: Component = () => {
    return <>
        <Show when={!isLoading()} fallback={"Loading..."}>
            <Home />
        </Show>
    </>
};

export default App;
