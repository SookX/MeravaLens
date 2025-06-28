import { Box, Typography } from "@mui/material"
import { Canvas, useFrame } from '@react-three/fiber'
import EarthScene from "./components/EarthScene/EarthScene"
import { PerspectiveCamera } from "@react-three/drei"

const Home = () => {
    return (
        <Canvas
            style={{ height: "100vh" }}
            camera={{
                fov: 20,
                position: [0, 0, 20],
                near: 0.1,
                far: 2000
            }}
        >
            <EarthScene />
        </Canvas>
    )
}

export default Home