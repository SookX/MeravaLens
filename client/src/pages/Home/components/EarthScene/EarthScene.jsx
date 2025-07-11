import { Canvas, useFrame } from '@react-three/fiber'
import { useEffect, useRef, useState } from "react"
import { useGLTF } from "@react-three/drei"
import { theme } from '../../../../theme/theme'

useGLTF.preload("/models/earth/earth.glb")

const EarthScene = () => {
    const model = useGLTF("/models/earth/earth.glb")

    const earthRef = useRef()
    
    const [earthPosition, setEarthPosition] = useState([2, -1, -5])


    const handleEarthReposition = (pos) => {
        if(earthPosition != pos) { setEarthPosition([...pos]) }
    }


    useFrame((state, delta) => {
        if(window.innerWidth > theme.breakpoints.values.md) earthRef.current.rotation.y += delta / 15
        else earthRef.current.rotation.x -= delta / 15

        if(window.innerWidth > theme.breakpoints.values.lg) {
            handleEarthReposition([2, -1, -5])
        }
        else if(window.innerWidth > theme.breakpoints.values.md) {
            handleEarthReposition([4, -1, -5])
        }
        else if(window.innerWidth > theme.breakpoints.values.xs) {
            handleEarthReposition([0, -3, -5])
        }
    })

    // useEffect(() => {
    //     const handleResize = () => {
    //         console.log('resiziing')
    //         if(window.innerWidth > theme.breakpoints.values.lg) setEarthPosition([2, -1, -5])
    //         // else if(window.innerWidth > theme.breakpoints.values.md) setEarthPosition([3, -1, -5])
    //         else if(window.innerWidth > theme.breakpoints.values.md) setEarthPosition([0, 0, 0])
    //     }

    //     handleResize()
    //     window.addEventListener('resize', handleResize())

    //     return window.removeEventListener('resize', handleResize())
    // }, [])

    return (
        <>
            <primitive
                ref={earthRef}
                object={model.scene}
                position={earthPosition}
                scale={1}
                rotation={[0, Math.PI / 4, 0]}
            />
            <pointLight intensity={200} position={[earthPosition[0] - 9, earthPosition[1] + 8, earthPosition[2] + 3]} />
        </>
    )
}

const EarthCanva = () => {
    return (
        <Canvas
            style={{ height: "100vh" }}
            camera={{
                fov: 20,
                position: [0, 0, 15],
                near: 0.1,
                far: 2000
            }}
        >
            <EarthScene />
        </Canvas>
    )
}

export default EarthCanva