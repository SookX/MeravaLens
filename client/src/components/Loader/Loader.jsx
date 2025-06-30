import { Backdrop } from '@mui/material'
import './loader.css'

const Loader = () => {
    return (
        <Backdrop sx={(theme) => ({ backgroundColor: "black", zIndex: theme.zIndex.drawer + 1 })} open={true} onClose={() => {}}>
            <div class="loader"></div>
        </Backdrop>
    )
}

export default Loader