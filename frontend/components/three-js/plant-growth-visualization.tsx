"use client"

import { useRef, useEffect, useState } from "react"
import * as THREE from "three"
import { Canvas, useFrame } from "@react-three/fiber"
import { Environment, OrbitControls } from "@react-three/drei"
import { Suspense } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, RefreshCw } from "lucide-react"

interface PlantGrowthVisualizationProps {
  height?: number
  className?: string
}

export default function PlantGrowthVisualization({ height = 100, className = "" }: PlantGrowthVisualizationProps) {
  const [growthStage, setGrowthStage] = useState(0.5) // 0 to 1
  const [isPlaying, setIsPlaying] = useState(false)
  const [autoRotate, setAutoRotate] = useState(true)

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setGrowthStage((prev) => {
          const newValue = prev + 0.01
          if (newValue >= 1) {
            setIsPlaying(false)
            return 1
          }
          return newValue
        })
      }, 100)
      return () => clearInterval(interval)
    }
  }, [isPlaying])

  const resetGrowth = () => {
    setGrowthStage(0)
    setIsPlaying(true)
  }

  return (
    <div className={`relative rounded-lg overflow-hidden bg-gradient-to-b from-lime-50 to-lime-100 ${className}`}>
      <div className="absolute top-4 left-4 z-10 bg-white/80 backdrop-blur-sm rounded-lg p-3 shadow-md">
        <h3 className="text-sm font-medium text-gray-800 mb-2">Plant Growth Visualization</h3>
        <p className="text-xs text-gray-600 mb-4">Current height: {Math.round(height * growthStage)} cm</p>

        <div className="space-y-4">
          <Slider
            value={[growthStage * 100]}
            min={0}
            max={100}
            step={1}
            onValueChange={(value) => {
              setGrowthStage(value[0] / 100)
              setIsPlaying(false)
            }}
            className="w-full"
          />

          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="flex-1" onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button variant="outline" size="sm" className="flex-1" onClick={resetGrowth}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="h-[400px]">
        <Canvas shadows dpr={[1, 2]}>
          <Suspense fallback={null}>
            <Scene growthStage={growthStage} autoRotate={autoRotate} />
            <Environment preset="sunset" />
          </Suspense>
        </Canvas>
      </div>
    </div>
  )
}

function Scene({ growthStage, autoRotate }: { growthStage: number; autoRotate: boolean }) {
  return (
    <>
      <color attach="background" args={["#f7fee7"]} />
      <fog attach="fog" args={["#f7fee7", 5, 20]} />

      <ambientLight intensity={0.5} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={1}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />

      <OrbitControls
        autoRotate={autoRotate}
        autoRotateSpeed={1}
        enableZoom={true}
        maxPolarAngle={Math.PI / 2}
        minPolarAngle={0}
      />

      <group position={[0, -1, 0]}>
        <Terrain />
        <Plant growthStage={growthStage} />
      </group>
    </>
  )
}

function Terrain() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <circleGeometry args={[3, 32]} />
      <meshStandardMaterial color="#84cc16" />
    </mesh>
  )
}

function Plant({ growthStage }: { growthStage: number }) {
  const group = useRef<THREE.Group>(null)

  // Calculate the current height based on growth stage
  const currentHeight = 3 * growthStage
  const stemSegments = 10
  const leafCount = Math.floor(4 * growthStage) + 1

  useFrame((state) => {
    if (group.current) {
      // Subtle swaying motion
      group.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.05
    }
  })

  return (
    <group ref={group}>
      {/* Plant pot */}
      <mesh position={[0, 0.3, 0]} castShadow>
        <cylinderGeometry args={[0.5, 0.3, 0.6, 16]} />
        <meshStandardMaterial color="#a16207" />
      </mesh>

      {/* Soil */}
      <mesh position={[0, 0.5, 0]} castShadow>
        <cylinderGeometry args={[0.45, 0.45, 0.2, 16]} />
        <meshStandardMaterial color="#422006" />
      </mesh>

      {/* Stem */}
      <mesh position={[0, 0.5 + currentHeight / 2, 0]} castShadow>
        <cylinderGeometry args={[0.05, 0.1, currentHeight, 8]} />
        <meshStandardMaterial color="#65a30d" />
      </mesh>

      {/* Leaves */}
      {Array.from({ length: leafCount }).map((_, i) => {
        const leafHeight = 0.6 + (currentHeight / leafCount) * i
        const leafSize = 0.2 + (i / leafCount) * 0.3
        const angle = ((i % 2 === 0 ? 1 : -1) * Math.PI) / 4

        return (
          <group key={i} position={[0, leafHeight, 0]} rotation={[0, (i * Math.PI) / 2, angle]}>
            <mesh castShadow>
              <coneGeometry args={[leafSize, leafSize * 2, 4]} />
              <meshStandardMaterial color="#84cc16" side={THREE.DoubleSide} />
            </mesh>
          </group>
        )
      })}

      {/* Flower/fruit - only appears in later stages */}
      {growthStage > 0.7 && (
        <group position={[0, 0.5 + currentHeight, 0]}>
          <mesh castShadow>
            <sphereGeometry args={[(0.2 * (growthStage - 0.7)) / 0.3, 16, 16]} />
            <meshStandardMaterial color="#facc15" />
          </mesh>
          {growthStage > 0.85 && (
            <>
              <mesh position={[0.1, 0.1, 0]} castShadow>
                <sphereGeometry args={[(0.1 * (growthStage - 0.85)) / 0.15, 16, 16]} />
                <meshStandardMaterial color="#facc15" />
              </mesh>
              <mesh position={[-0.1, 0.05, 0.05]} castShadow>
                <sphereGeometry args={[(0.08 * (growthStage - 0.85)) / 0.15, 16, 16]} />
                <meshStandardMaterial color="#facc15" />
              </mesh>
            </>
          )}
        </group>
      )}
    </group>
  )
}

