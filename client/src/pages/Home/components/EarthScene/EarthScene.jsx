import { useFrame } from '@react-three/fiber'
import { useRef } from "react"
import { useGLTF } from "@react-three/drei"

useGLTF.preload("/models/earth/earth.glb")

const EarthScene = () => {
    const model = useGLTF("/models/earth/earth.glb")

    const earthRef = useRef()
    
    useFrame((state, delta) => {
        // earthRef.current.rotation.x -= delta / 8
        earthRef.current.rotation.y += delta / 15
    })

    return (
        <>
            <primitive
                ref={earthRef}
                object={model.scene}
                position={[7, -1, -5]}
                scale={1}
                rotation={[0, Math.PI / 4, 0]}
            />
            <pointLight intensity={200} position={[-2, 7, -2]} />
        </>
    )
}

export default EarthScene