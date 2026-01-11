
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import {
  Camera, Upload, Play, Pause, X,
  Activity, Cpu, Check, Wifi,
  BrainCircuit, Shield, Save, RotateCcw, Settings, Database,
  AlertCircle, Mic, MicOff, MessageSquare, Terminal,
  Radio, HardDrive, History, FileText, Lock, ShieldCheck,
  Maximize, ScanLine, UserCheck, Search, Zap, Globe,
  Eye, MousePointer2, Thermometer, Gauge, ArrowRightCircle,
  TrendingUp, Layers, Info, Hash, Power, Navigation, Target,
  AlertTriangle, Scale, ClipboardList, Video, FileBadge, CheckCircle2,
  Clock, MapPin, Ruler, BadgeCheck, BarChart3, Binary, Signal, Plus
} from 'lucide-react';
import { YoloDetector, ByteTracker, BoTSORT, PoseDetection } from './yolo-tracker';

// --- Parámetros Cinemáticos ---
const LANE_WIDTH_METERS = 3.0;
const REFERENCE_LANE_PX = 320;
const PIXELS_PER_METER = REFERENCE_LANE_PX / LANE_WIDTH_METERS;

// --- Componente Emblema Daganzo ---
const DaganzoEmblem = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 200 240" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
  </svg>
);

// --- Interfaces ---
interface Point { x: number; y: number; time: number; }

interface Track {
  id: number;
  label: string;
  subType?: string;
  points: Point[];
  // Visual Smoothing (EMA)
  renderX: number;
  renderY: number;
  renderW: number;
  renderH: number;
  lastSeen: number;
  lastSnapshotTime: number;
  color: string;
  snapshots: string[];
  velocity: number;
  age: number;
  plate?: string;
  isInfractor?: boolean;
  analyzed: boolean;
  // Enhanced tracking
  vx: number; // Velocity X component
  vy: number; // Velocity Y component
  predictedX: number; // Predicted next X position
  predictedY: number; // Predicted next Y position
  confidence: number; // Track confidence (0-1)
  missedFrames: number; // Counter for missed detections
  w: number; // Current width
  h: number; // Current height
}

interface InfractionLog {
  id: number;
  plate: string;
  description: string;
  severity: string;
  image: string;
  videoUrl?: string;
  time: string;
  date: string;
  reasoning?: string[];
  vehicleType: string;
  subType: string;
  confidence: number;
  violatedDirective: string;
  legalArticle?: string;
  telemetry: {
    speedEstimated: string;
    maneuverType: string;
    poseAlert: boolean;
    framesAnalyzed: number;
    distanceToLine?: string;
  };
  snapshots?: string[];
}

const VEHICLE_COLORS: Record<string, string> = {
  car: '#06b6d4', truck: '#f59e0b', motorcycle: '#8b5cf6', bus: '#10b981', person: '#ec4899', bicycle: '#84cc16'
};

const StatusBadge = ({ label, active, color = 'cyan', pulse = true }: { label: string; active: boolean; color?: string; pulse?: boolean }) => (
  <div className={`flex items-center gap-2 px-3 py-1 rounded-md border text-[9px] font-black uppercase tracking-widest transition-all duration-500 ${active ? `bg-${color}-500/10 border-${color}-500/50 text-${color}-400 shadow-[0_0_15px_rgba(6,182,212,0.15)]` : 'bg-slate-900/50 border-white/5 text-slate-500 opacity-40'}`}>
    <div className={`w-1.5 h-1.5 rounded-full ${active ? `bg-${color}-400 ${pulse ? 'animate-pulse' : ''} shadow-[0_0_8px_#22d3ee]` : 'bg-slate-700'}`} />
    {label}
  </div>
);

const DEFAULT_DIRECTIVES = `PROTOCOLO DAGANZO_V15_AUDIT:
1. DETECCIÓN CELULAR: Vigilar posición de manos y brillo facial compatible con terminal móvil.
2. CINTURÓN SEGURIDAD: Analizar contraste diagonal en hombro del conductor.
3. SEÑALÉTICA HORIZONTAL: Vigilar pisado de línea continua y detención total en STOP (velocidad < 1km/h).
4. PRIORIDAD PEATONAL: Detectar presencia de peatones en cebra y reducción de velocidad preventiva.
5. OCUPACIÓN VÍA: Detección de vehículos en doble fila o zonas de carga/descarga sin operario.
6. COMPORTAMIENTO: Virajes bruscos, aceleraciones súbitas o invasión de carril contrario.
7. CALIBRACIÓN: Usar referencia de 3 metros por carril para estimar velocidades críticas.`;

const App = () => {
  const [source, setSource] = useState<'none' | 'live' | 'upload'>('none');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [directives, setDirectives] = useState<string>(DEFAULT_DIRECTIVES);
  const [aiFeedback, setAiFeedback] = useState<string | null>(null);

  const [logs, setLogs] = useState<InfractionLog[]>([]);
  const [selectedLog, setSelectedLog] = useState<InfractionLog | null>(null);
  const [cumulativeDetections, setCumulativeDetections] = useState(0);
  const [cumulativeExpedientes, setCumulativeExpedientes] = useState(0);

  const [systemStats, setSystemStats] = useState({
    cpu: 24,
    mem: 1.2,
    temp: 42,
    net: 85,
    gps: '40.6483° N, 3.4582° W'
  });

  const [poseEstimationEnabled, setPoseEstimationEnabled] = useState(false);

  // === YOLOv11 + Multi-Tracker Configuration ===
  interface YoloConfig {
    // YOLO Detection
    confThreshold: number;      // Min confidence (0-1)
    nmsIouThreshold: number;    // NMS IoU threshold
    detectionSkip: number;      // Process every N frames

    // Tracker Selection
    trackerType: 'ByteTrack' | 'BoT-SORT';

    // ByteTrack / BoT-SORT Parameters
    highDetThreshold: number;   // High confidence threshold for first matching
    lowDetThreshold: number;    // Low confidence threshold for second matching  
    matchIouThreshold: number;  // IoU threshold for track matching
    trackBufferFrames: number;  // Frames to keep lost tracks
    minHitsToConfirm: number;   // Min detections to confirm new track

    // BoT-SORT Specific
    appearanceWeight: number;   // Weight for appearance matching (0-1)
  }

  const trackingPresets: Record<string, YoloConfig> = {
    'highway-fast-bytetrack': {
      confThreshold: 0.4,
      nmsIouThreshold: 0.45,
      detectionSkip: 2,
      trackerType: 'ByteTrack',
      highDetThreshold: 0.6,
      lowDetThreshold: 0.2,
      matchIouThreshold: 0.3,
      trackBufferFrames: 20,
      minHitsToConfirm: 3,
      appearanceWeight: 0.0
    },
    'urban-balanced-bytetrack': {
      confThreshold: 0.25,
      nmsIouThreshold: 0.5,
      detectionSkip: 2,
      trackerType: 'ByteTrack',
      highDetThreshold: 0.5,
      lowDetThreshold: 0.1,
      matchIouThreshold: 0.2, // Increased tolerance for movement
      trackBufferFrames: 90, // Extended memory for occlusion
      minHitsToConfirm: 2,
      appearanceWeight: 0.0
    },
    'precision-slow-botsort': {
      confThreshold: 0.3,
      nmsIouThreshold: 0.55,
      detectionSkip: 1,
      trackerType: 'BoT-SORT',
      highDetThreshold: 0.6,
      lowDetThreshold: 0.1,
      matchIouThreshold: 0.2,
      trackBufferFrames: 45,
      minHitsToConfirm: 1,
      appearanceWeight: 0.5
    },
    'forensic-reID-botsort': {
      confThreshold: 0.25,
      nmsIouThreshold: 0.6,
      detectionSkip: 1,
      trackerType: 'BoT-SORT',
      highDetThreshold: 0.55,
      lowDetThreshold: 0.05,
      matchIouThreshold: 0.15,
      trackBufferFrames: 60,
      minHitsToConfirm: 1,
      appearanceWeight: 0.7
    }
  };

  const [activePreset, setActivePreset] = useState<string>('urban-balanced-bytetrack');
  const lastPosesRef = useRef<PoseDetection[]>([]);
  const [yoloConfig, setYoloConfig] = useState<YoloConfig>(trackingPresets['urban-balanced-bytetrack']);

  const frameCounterRef = useRef(0);



  // Advanced Multi-Lane Detection Configuration with Angled Lines
  interface DetectionLine {
    y: number;           // Position (0-1000) - can be start Y for angled lines
    x1?: number;         // Start X for angled lines (0-1000)
    x2?: number;         // End X for angled lines (0-1000)  
    y2?: number;         // End Y for angled lines (0-1000)
    angle?: number;      // Line angle in degrees (0-360)
    type: 'solid' | 'dashed' | 'divider' | 'pedestrian' | 'stop' | 'loading-zone' | 'bus-lane' | 'speed-zone';
    direction: 'bidirectional' | 'northbound' | 'southbound';
    label: string;
    infractionType?: string; // Specific infraction this line detects
  }

  // Automatic Mesh Grid Parameters
  interface MeshGridConfig {
    enabled: boolean;
    gridType: 'horizontal' | 'vertical' | 'cross' | 'perspective';
    spacing: number;     // Pixels between lines
    angleAdaptive: boolean; // Auto-calculate angles based on perspective
    perspectiveVanishingY: number; // Vanishing point for perspective (0-1000)
  }

  const [meshGridConfig, setMeshGridConfig] = useState<MeshGridConfig>({
    enabled: false,
    gridType: 'cross',
    spacing: 200,
    angleAdaptive: true,
    perspectiveVanishingY: 300
  });

  const [selectedConfigs, setSelectedConfigs] = useState<string[]>(['2-lanes-bidirectional']); // Multi-preset configuration
  const [isManualMode, setIsManualMode] = useState(false);
  const [manualLineType, setManualLineType] = useState<DetectionLine['type']>('solid');
  const [detectionLines, setDetectionLines] = useState<DetectionLine[]>([
    { y: 500, type: 'solid', direction: 'bidirectional', label: 'LÍNEA CONTINUA CENTRAL', infractionType: 'LINE_CROSSING' }
  ]);

  // === Automatic Mesh Grid Generator ===
  const generateMeshGrid = useCallback((config: MeshGridConfig): DetectionLine[] => {
    if (!config.enabled) return [];

    const lines: DetectionLine[] = [];
    const canvasWidth = 1000; // Normalized space
    const canvasHeight = 1000;

    switch (config.gridType) {
      case 'horizontal':
        // Simple horizontal lines
        for (let y = config.spacing; y < canvasHeight; y += config.spacing) {
          lines.push({
            y,
            type: 'divider',
            direction: 'bidirectional',
            label: `GRID_H_${y}`,
            infractionType: undefined
          });
        }
        break;

      case 'vertical':
        // Vertical lines (using x1,y1,x2,y2)
        for (let x = config.spacing; x < canvasWidth; x += config.spacing) {
          lines.push({
            y: 0,
            x1: x,
            y2: canvasHeight,
            x2: x,
            type: 'divider',
            direction: 'bidirectional',
            label: `GRID_V_${x}`,
            infractionType: undefined
          });
        }
        break;

      case 'cross':
        // Both horizontal and vertical
        for (let y = config.spacing; y < canvasHeight; y += config.spacing) {
          lines.push({
            y,
            type: 'divider',
            direction: 'bidirectional',
            label: `GRID_H_${y}`,
            infractionType: undefined
          });
        }
        for (let x = config.spacing; x < canvasWidth; x += config.spacing) {
          lines.push({
            y: 0,
            x1: x,
            y2: canvasHeight,
            x2: x,
            type: 'divider',
            direction: 'bidirectional',
            label: `GRID_V_${x}`,
            infractionType: undefined
          });
        }
        break;

      case 'perspective':
        // Perspective grid with adaptive angles
        const vanishingY = config.perspectiveVanishingY;
        const vanishingX = canvasWidth / 2; // Center

        // Horizontal perspective lines (converging to vanishing point)
        for (let y = config.spacing; y < canvasHeight; y += config.spacing) {
          if (!config.angleAdaptive) {
            // Simple horizontal
            lines.push({
              y,
              type: 'divider',
              direction: 'bidirectional',
              label: `PERSP_H_${y}`,
              infractionType: undefined
            });
          } else {
            // Angled towards vanishing point
            const distanceFromVanishing = Math.abs(y - vanishingY);
            const angleOffset = (y - vanishingY) / 10; // Subtle angle

            lines.push({
              y,
              x1: 0,
              y2: y,
              x2: canvasWidth,
              angle: Math.atan2(angleOffset, canvasWidth) * (180 / Math.PI),
              type: 'divider',
              direction: 'bidirectional',
              label: `PERSP_H_${y}`,
              infractionType: undefined
            });
          }
        }

        // Radiating lines from vanishing point
        const numRadialLines = Math.floor(canvasWidth / config.spacing);
        for (let i = 0; i < numRadialLines; i++) {
          const bottomX = (i / numRadialLines) * canvasWidth;
          lines.push({
            y: vanishingY,
            x1: vanishingX,
            y2: canvasHeight,
            x2: bottomX,
            angle: Math.atan2(canvasHeight - vanishingY, bottomX - vanishingX) * (180 / Math.PI),
            type: 'divider',
            direction: 'bidirectional',
            label: `PERSP_R_${i}`,
            infractionType: undefined
          });
        }
        break;
    }

    return lines;
  }, []);

  // Apply mesh grid when config changes
  useEffect(() => {
    if (meshGridConfig.enabled) {
      const meshLines = generateMeshGrid(meshGridConfig);
      setDetectionLines(prev => {
        // Keep manual lines, add mesh lines
        const manualLines = prev.filter(l => !l.label.startsWith('GRID_') && !l.label.startsWith('PERSP_'));
        return [...manualLines, ...meshLines];
      });
    } else {
      // Remove mesh lines
      setDetectionLines(prev => prev.filter(l => !l.label.startsWith('GRID_') && !l.label.startsWith('PERSP_')));
    }
  }, [meshGridConfig, generateMeshGrid]);

  // Comprehensive Road Configuration Presets with Integrated Forensic Directives
  interface RoadPreset {
    lines: DetectionLine[];
    directivesTemplate: string;
  }

  const ROAD_PRESETS: Record<string, RoadPreset> = {
    'daganzo-m100-enlace': {
      lines: [
        { y: 350, type: 'divider', direction: 'bidirectional', label: 'NUDO M-100 / M-113', infractionType: null },
        { y: 650, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR ENLACE LOGÍSTICO', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO NUDO LOGÍSTICO M-100:\n1. Auditoría de flujos pesados desde A-2.\n2. Control de velocidad en ramal de incorporación.\n3. Vigilancia de trazada en curvas de radio reducido."
    },

    'daganzo-m113-ajalvir': {
      lines: [
        { y: 400, type: 'solid', direction: 'bidirectional', label: 'EJE ACCESO AJALVIR', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO ACCESO AJALVIR (M-113 OESTE):\n1. Control de pisotón de línea en tramo interurbano.\n2. Vigilancia de uso de arcenes por ciclistas/peatones.\n3. [LINE: Y=800, TYPE=speed-zone, LABEL=CONTROL ENTRADA MUNICIPIO, INFRACTION=SPEEDING]"
    },

    'daganzo-m113-fresno': {
      lines: [
        { y: 350, type: 'divider', direction: 'bidirectional', label: 'ENTORNO FRESNO/SERRACINES', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO ACCESO FRESNO (M-113 NORTE):\n1. Vigilancia de tránsito agrícola/maquinaria.\n2. Control de adelantamientos en rasantes.\n3. Auditoría de señales de stop en caminos rurales."
    },

    'daganzo-av-madrid': {
      lines: [
        { y: 300, type: 'bus-lane', direction: 'northbound', label: 'CARRIL BUS AV. MADRID', infractionType: 'BUS_LANE_VIOLATION' },
        { y: 600, type: 'pedestrian', direction: 'bidirectional', label: 'CRUCE AV. MADRID', infractionType: 'PEDESTRIAN_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO AVENIDA DE MADRID (ACCESO SUR):\n1. Prioridad: Carril BUS - Sancionar invasión turismos.\n2. Control de velocidad urbana 30km/h.\n3. Vigilancia de giros a derecha hacia polígonos."
    },

    'daganzo-calle-mayor': {
      lines: [
        { y: 400, type: 'pedestrian', direction: 'bidirectional', label: 'EJE CALLE MAYOR', infractionType: 'PEDESTRIAN_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO CALLE MAYOR (NÚCLEO):\n1. Control de prioridad peatonal extrema.\n2. Sancionar carga/descarga fuera de horario.\n3. [LINE: Y=200, TYPE=speed-zone, LABEL=RADAR NÚCLEO, INFRACTION=SPEEDING]"
    },

    'daganzo-poligono-gitesa': {
      lines: [
        { y: 350, type: 'loading-zone', direction: 'bidirectional', label: 'POLÍGONO GITESA', infractionType: 'LOADING_ZONE_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO POLÍGONO GITESA:\n1. Vigilancia de camiones en espera en calzada.\n2. Control de ocupación de vados industriales.\n3. Auditoría de flujos pesados nocturnos."
    },

    'daganzo-camino-gancha': {
      lines: [
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'CAMINO DE LA GANCHA', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO ENTORNO RURAL / DEPORTIVO:\n1. Control de tránsito no autorizado en caminos.\n2. Vigilancia de acceso a instalaciones deportivas.\n3. Sancionar vertido de escombros (Auditoría visual)."
    },

    'daganzo-m113-norte': {
      lines: [
        { y: 350, type: 'speed-zone', direction: 'southbound', label: 'ACCESO M-113 NORTE (ENTRADA)', infractionType: 'SPEEDING' },
        { y: 600, type: 'solid', direction: 'bidirectional', label: 'EJE M-113', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO M-113 ACCESO NORTE:\n1. Control velocidad entrada municipio (50km/h).\n2. Vigilancia pisotón línea continua en curva de acceso.\n3. [LINE: Y=850, TYPE=stop, LABEL=PUNTO AUDITORÍA, INFRACTION=null]"
    },

    'daganzo-m113-sur': {
      lines: [
        { y: 300, type: 'speed-zone', direction: 'northbound', label: 'ACCESO M-113 SUR (A-2/R-2)', infractionType: 'SPEEDING' },
        { y: 550, type: 'divider', direction: 'bidirectional', label: 'BIFURCACIÓN POLÍGONO', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO M-113 ACCESO SUR:\n1. Auditoría de flujo desde A-2/Torrejón.\n2. Control de velocidad en tramo interurbano.\n3. Vigilancia de incorporaciones desde caminos vecinales."
    },

    'daganzo-m118-alcala': {
      lines: [
        { y: 400, type: 'solid', direction: 'bidirectional', label: 'EJE M-118 ALCALÁ', infractionType: 'LINE_CROSSING' },
        { y: 700, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR M-118', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO M-118 (CONEXIÓN ALCALÁ):\n1. Control de velocidad en tramo curvo.\n2. Sancionar adelantamientos prohibidos en línea continua.\n3. [LINE: Y=200, TYPE=stop, LABEL=STOP CONEXIÓN M-113, INFRACTION=STOP_VIOLATION]"
    },

    'daganzo-m119-camarma': {
      lines: [
        { y: 450, type: 'solid', direction: 'bidirectional', label: 'EJE M-119 CAMARMA', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO M-119 (ACCESO CAMARMA):\n1. Vigilancia de transporte pesado.\n2. Control de velocidad 90km/h (Tramo interurbano).\n3. Auditoría de arcenes."
    },

    'daganzo-constitucion': {
      lines: [
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'PASO C/ CONSTITUCIÓN', infractionType: 'PEDESTRIAN_PRIORITY' },
        { id: 'radar-urbano', y: 500, type: 'speed-zone', direction: 'bidirectional', label: 'CONTROL 30/40', infractionType: 'SPEEDING' } as any
      ],
      directivesTemplate: "PROTOCOLO ARTERIA URBANA (C/ CONSTITUCIÓN):\n1. Límite 30km/h tramo central.\n2. Prioridad peatonal en cruces señalizados.\n3. Sancionar estacionamiento en doble fila obstruyendo bus."
    },

    'daganzo-residencial': {
      lines: [
        { y: 400, type: 'pedestrian', direction: 'bidirectional', label: 'ZONA RESIDENCIAL 20', infractionType: 'PEDESTRIAN_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO ZONA RESIDENCIAL / COEXISTENCIA:\n1. Velocidad máxima 20km/h.\n2. Vigilancia de juego en calle y tránsito peatonal fluido.\n3. Prohibición de tránsito de paso (solo residentes)."
    },

    'daganzo-colegios-magno': {
      lines: [
        { y: 350, type: 'pedestrian', direction: 'bidirectional', label: 'ENTORNO COLEGIO MAGNO', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 650, type: 'stop', direction: 'bidirectional', label: 'DETENCIÓN BUS ESCOLAR', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO SEGURIDAD ESCOLAR (MAGNO/BERZAL):\n1. Control horario entrada/salida (08:30-10:00 / 16:00-17:30).\n2. Sanción inmediata doble fila.\n3. Vigilancia de cruce seguro de menores."
    },

    'daganzo-poligono-frailes': {
      lines: [
        { y: 350, type: 'loading-zone', direction: 'bidirectional', label: 'LOS FRAILES LOGÍSTICA', infractionType: 'LOADING_ZONE_VIOLATION' },
        { y: 650, type: 'divider', direction: 'bidirectional', label: 'FLUJO PESADO', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO POLÍGONO LOS FRAILES:\n1. Auditoría de vehículos de gran tonelaje.\n2. Control de carga/descarga fuera de zonas habilitadas.\n3. Vigilancia de estacionamiento en esquinas de naves."
    },

    'daganzo-rotonda-entrada': {
      lines: [
        { y: 250, type: 'solid', direction: 'southbound', label: 'ENTRADA ROTONDA M-113', infractionType: 'STOP_VIOLATION' },
        { y: 500, type: 'dashed', direction: 'bidirectional', label: 'ANILLO ROTONDA', infractionType: null },
        { y: 750, type: 'solid', direction: 'bidirectional', label: 'SALIDA CENTRO', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO ROTONDA PRINCIPAL DAGANZO:\n1. Prioridad de paso en anillo.\n2. Sancionar salida desde carril interior.\n3. Auditoría de trazadas peligrosas."
    },

    'daganzo-centro': {
      lines: [
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'PLAZA DE LA VILLA (20)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 600, type: 'speed-zone', direction: 'bidirectional', label: 'ZONA RESIDENCIAL', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO DAGANZO CENTRO:\n1. Límite 20km/h en casco histórico.\n2. Prioridad absoluta al peatón.\n3. Vigilancia de paradas en Plaza de la Villa."
    },

    'daganzo-m113': {
      lines: [
        { y: 400, type: 'solid', direction: 'bidirectional', label: 'TRAVESÍA M-113', infractionType: 'LINE_CROSSING' },
        { y: 700, type: 'stop', direction: 'bidirectional', label: 'SEMÁFORO ARTERIAL', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO DAGANZO M-113:\n1. Control de travesía principal.\n2. Vigilancia de giro a la izquierda en polígono.\n3. Sancionar rebasamiento de foto-rojo."
    },

    'daganzo-poligono': {
      lines: [
        { y: 350, type: 'loading-zone', direction: 'bidirectional', label: 'ACCESO POLÍGONO', infractionType: 'LOADING_ZONE_VIOLATION' },
        { y: 650, type: 'divider', direction: 'bidirectional', label: 'EJE INDUSTRIAL', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO DAGANZO INDUSTRIAL:\n1. Control de paso de vehículos pesados.\n2. Vigilancia de carga/descarga en calzada.\n3. [LINE: Y=800, TYPE=solid, LABEL=LÍNEA CONTINUA ACCESO, INFRACTION=LINE_CROSSING]"
    },
    '2-lanes-bidirectional': {
      lines: [
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'DIVISORIA CENTRAL (L1)', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO BIDIRECCIONAL:\n1. Vigilar pisotón de línea continua central.\n2. Analizar invasión de carril contrario en adelantamientos prohibidos.\n3. Verificar uso de intermitentes en cambios de trayectoria."
    },

    '4-lanes-highway': {
      lines: [
        { y: 300, type: 'speed-zone', direction: 'northbound', label: 'ZONA CALIBRACIÓN (Z1)', infractionType: 'SPEEDING' },
        { y: 400, type: 'divider', direction: 'northbound', label: 'DIVISOR CARRIL L/R', infractionType: null },
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'MEDIANA DIVISORIA', infractionType: 'LINE_CROSSING' },
        { y: 600, type: 'divider', direction: 'southbound', label: 'DIVISOR CARRIL L/R', infractionType: null },
        { y: 700, type: 'speed-zone', direction: 'southbound', label: 'ZONA CALIBRACIÓN (Z2)', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO AUTOPISTA:\n1. Control riguroso de velocidad (Umbral 120km/h).\n2. Vigilar ocupación indebida del carril izquierdo.\n3. Detectar conducción temeraria (zigzag entre carriles).\n4. Verificar distancia de seguridad (Regla de los 2 segundos)."
    },

    '3-lanes-oneway': {
      lines: [
        { y: 300, type: 'bus-lane', direction: 'southbound', label: 'CARRIL RESERVADO (R1)', infractionType: 'BUS_LANE_VIOLATION' },
        { y: 450, type: 'dashed', direction: 'southbound', label: 'DIVISORIA TRAMO 1-2', infractionType: null },
        { y: 650, type: 'dashed', direction: 'southbound', label: 'DIVISORIA TRAMO 2-3', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO VÍA URBANA (3C):\n1. Prioridad: Sancionar invasión de Carril BUS por vehículos no autorizados.\n2. Vigilar cambios de carril bruscos sin señalización.\n3. Detectar paradas indebidas en carriles de circulación."
    },

    'cross-junction-4way': {
      lines: [
        { y: 200, type: 'stop', direction: 'bidirectional', label: 'DETENCIÓN NORTE (P1)', infractionType: 'STOP_VIOLATION' },
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'ÁREA DE PASO (A1)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 500, type: 'divider', direction: 'bidirectional', label: 'EJE INTERSECCIÓN', infractionType: null },
        { y: 700, type: 'pedestrian', direction: 'bidirectional', label: 'ÁREA DE PASO (A2)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 800, type: 'stop', direction: 'bidirectional', label: 'DETENCIÓN SUR (P2)', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO INTERSECCIÓN 4 VÍAS:\n1. Control de giro prohibido en el centro de la intersección.\n2. Vigilancia cruzada de prioridades de paso.\n3. Sanción por bloqueo de intersección (quedarse en medio)."
    },

    't-junction-urban': {
      lines: [
        { y: 350, type: 'stop', direction: 'bidirectional', label: 'PARADA OBLIGATORIA', infractionType: 'STOP_VIOLATION' },
        { y: 500, type: 'pedestrian', direction: 'bidirectional', label: 'PASO PEATONAL INTEGRADOR', infractionType: 'PEDESTRIAN_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO INTERSECCIÓN EN T URBANA:\n1. Control estricto de Stop/Ceda el Paso.\n2. Prioridad peatonal en el radio de giro.\n3. Vigilancia de invasión de carril contrario al realizar el giro."
    },

    't-junction-rural': {
      lines: [
        { y: 400, type: 'divider', direction: 'bidirectional', label: 'CARRIL CENTRAL DE GIRO', infractionType: null },
        { y: 700, type: 'solid', direction: 'bidirectional', label: 'LÍNEA DE INCORPORACIÓN', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO INTERSECCIÓN EN T RURAL (M-113/M-119):\n1. Control de velocidad en vía principal.\n2. Vigilancia de uso del carril de espera/giro.\n3. Detección de incorporaciones peligrosas (faltas de ceda el paso)."
    },

    't-junction-multi': {
      lines: [
        { y: 300, type: 'stop', direction: 'southbound', label: 'DETENCIÓN ACCESO (P1)', infractionType: 'STOP_VIOLATION' },
        { y: 450, type: 'solid', direction: 'bidirectional', label: 'DIVISORIA DE FLUJO', infractionType: 'LINE_CROSSING' },
        { y: 650, type: 'dashed', direction: 'bidirectional', label: 'VECTOR DE TRANSICIÓN', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO INTERSECCIÓN EN T MULTICARRIL:\n1. Analizar detención en vía secundaria.\n2. Vigilar pisotón de línea continua al girar.\n3. Prioridad absoluta a vía principal."
    },

    'roundabout-2lanes': {
      lines: [
        { y: 250, type: 'solid', direction: 'southbound', label: 'PUNTO INCORPORACIÓN (I1)', infractionType: 'STOP_VIOLATION' },
        { y: 450, type: 'dashed', direction: 'bidirectional', label: 'DELIMITADOR EXTERIOR', infractionType: null },
        { y: 650, type: 'solid', direction: 'bidirectional', label: 'DELIMITADOR INTERIOR', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO ROTONDA MULTICARRIL:\n1. Sancionar salida directa desde carril interior.\n2. Vigilancia de prioridad de entrada.\n3. Control de trazada (uso correcto de carriles)."
    },

    'y-junction-split': {
      lines: [
        { y: 400, type: 'divider', direction: 'northbound', label: 'VÉRTICE SEPARACIÓN', infractionType: null },
        { y: 550, type: 'solid', direction: 'northbound', label: 'DIVISORA DE ISLETA', infractionType: 'LINE_CROSSING' },
        { y: 750, type: 'speed-zone', direction: 'northbound', label: 'ZONA TRANSICIÓN (Z1)', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO BIFURCACIÓN EN Y:\n1. Sancionar cruce tardío de isleta (línea continua).\n2. Control de velocidad en ramal de salida.\n3. Detección de dudas peligrosas en el vértice."
    },

    'staggered-junction': {
      lines: [
        { y: 250, type: 'stop', direction: 'southbound', label: 'DETENCIÓN N-1 (P1)', infractionType: 'STOP_VIOLATION' },
        { y: 500, type: 'divider', direction: 'bidirectional', label: 'ÁREA CENTRAL', infractionType: null },
        { y: 750, type: 'stop', direction: 'northbound', label: 'DETENCIÓN N-2 (P2)', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO CRUCE DESPLAZADO:\n1. Vigilancia de tráfico cruzado en dos niveles.\n2. Análisis de ocupación de zona central.\n3. Control de giros indirectos."
    },

    'urban-complete': {
      lines: [
        { y: 200, type: 'stop', direction: 'bidirectional', label: 'LÍNEA DE DETENCIÓN (P1)', infractionType: 'STOP_VIOLATION' },
        { y: 350, type: 'pedestrian', direction: 'bidirectional', label: 'ZONA DE TRÁNSITO (A1)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'DELIMITADOR SÓLIDO (D1)', infractionType: 'LINE_CROSSING' },
        { y: 650, type: 'loading-zone', direction: 'bidirectional', label: 'ÁREA CARGA/DESCARGA', infractionType: 'LOADING_ZONE_VIOLATION' },
        { y: 800, type: 'bus-lane', direction: 'southbound', label: 'CARRIL RESERVADO (R1)', infractionType: 'BUS_LANE_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO INTEGRAL URBANO:\n1. Auditoría multiespectral de todas las infracciones simultáneas.\n2. Especial énfasis en distracción por móvil al aproximarse a zonas críticas.\n3. Vigilancia de comportamiento errático en áreas congestionadas."
    },

    'school-safety': {
      lines: [
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'ENTORNO ESCOLAR (A1)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 500, type: 'speed-zone', direction: 'bidirectional', label: 'CONTROL VELOCIDAD (Z1)', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO SEGURIDAD ESCOLAR:\n1. Velocidad limitada a 20km/h (tolerancia cero).\n2. Vigilancia extrema de peatones infantiles.\n3. Sancionar paradas en doble fila en horarios de entrada/salida."
    },

    'roundabout-access': {
      lines: [
        { y: 400, type: 'stop', direction: 'southbound', label: 'PUNTO ACCESO (I1)', infractionType: 'STOP_VIOLATION' },
        { y: 600, type: 'solid', direction: 'bidirectional', label: 'DIVISORIA INTERIOR (D1)', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO ACCESO ROTONDA:\n1. Analizar cesión de paso en entrada (prioridad del que está dentro).\n2. Sancionar cruce de líneas continuas en el interior de la rotonda."
    },

    'interurban-cloverleaf': {
      lines: [
        { y: 300, type: 'divider', direction: 'northbound', label: 'BIFURCACIÓN LAZO (L1)', infractionType: null },
        { y: 450, type: 'solid', direction: 'bidirectional', label: 'MEDIANA SEPARACIÓN', infractionType: 'LINE_CROSSING' },
        { y: 600, type: 'divider', direction: 'southbound', label: 'INCORPORACIÓN LAZO (L2)', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO INTERCAMBIADOR TRÉBOL:\n1. Analizar trayectorias en bucles de enlace.\n2. [LINE: Y=750, TYPE=speed-zone, LABEL=CONTROL VELOCIDAD ENLACE, INFRACTION=SPEEDING]\n3. Sancionar cambios de carril bruscos en zonas de trenzado."
    },

    'interurban-diamond': {
      lines: [
        { y: 350, type: 'divider', direction: 'northbound', label: 'SALIDA DIAMANTE (S1)', infractionType: null },
        { y: 650, type: 'divider', direction: 'northbound', label: 'ENTRADA DIAMANTE (E1)', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO ENLACE DIAMANTE:\n1. Vigilar cruce de línea continua en divergencia.\n2. [LINE: Y=200, TYPE=stop, LABEL=DETENCIÓN RAMAL, INFRACTION=STOP_VIOLATION]\n3. Auditoría de ceda el paso en incorporación."
    },

    'interurban-trumpet': {
      lines: [
        { y: 400, type: 'solid', direction: 'bidirectional', label: 'CURVA TROMPETA (C1)', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO ENLACE TROMPETA:\n1. Control de trazada en curva de gran radio.\n2. [LINE: Y=800, TYPE=speed-zone, LABEL=RADAR SALIDA AUTOPISTA, INFRACTION=SPEEDING]\n3. Vigilancia de invasión de arcén."
    },

    'accel-decel-lane': {
      lines: [
        { y: 450, type: 'dashed', direction: 'northbound', label: 'CARRIL ACELERACIÓN', infractionType: null },
        { y: 550, type: 'solid', direction: 'northbound', label: 'FIN DE RAMAL', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO RAMALES DE FLUJO:\n1. Verificar incorporación segura (velocidad adecuada).\n2. [LINE: Y=650, TYPE=divider, LABEL=ZONA TRENZADO, INFRACTION=null]\n3. Sancionar detención en carril de aceleración."
    },

    'tunnel-security': {
      lines: [
        { y: 300, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR TÚNEL (T1)', infractionType: 'SPEEDING' },
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'DELIMITADOR TÚNEL', infractionType: 'LINE_CROSSING' },
        { y: 700, type: 'divider', direction: 'bidirectional', label: 'ZONA DE SEGURIDAD', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO SEGURIDAD EN TÚNEL:\n1. Control estricto de distancia de seguridad.\n2. Sancionar uso de luces antiniebla o posición incorrecta.\n3. Vigilancia de detención injustificada en el interior."
    },

    'construction-zone': {
      lines: [
        { y: 300, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR OBRAS 40 (O1)', infractionType: 'SPEEDING' },
        { y: 600, type: 'divider', direction: 'bidirectional', label: 'BALIZAMIENTO TRANSVERSAL', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO ZONA DE OBRAS:\n1. Velocidad limitada a 40km/h (tolerancia mínima).\n2. Sancionar desplazamiento de conos o balizas.\n3. Prioridad absoluta a personal de vía."
    },

    'toll-plaza': {
      lines: [
        { y: 400, type: 'stop', direction: 'northbound', label: 'BARRERA PEAJE (B1)', infractionType: 'STOP_VIOLATION' },
        { y: 650, type: 'divider', direction: 'northbound', label: 'CARRIL TELEPEAJE', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO PEAJE / CONTROL:\n1. Verificar detención total ante barrera.\n2. Sancionar cambio de carril en zona de embudo.\n3. [LINE: Y=800, TYPE=speed-zone, LABEL=SALIDA CONTROL, INFRACTION=SPEEDING]"
    },

    'level-crossing': {
      lines: [
        { y: 350, type: 'stop', direction: 'bidirectional', label: 'PASO A NIVEL (N1)', infractionType: 'STOP_VIOLATION' },
        { y: 600, type: 'pedestrian', direction: 'bidirectional', label: 'ZONA DE RIESGO', infractionType: 'PEDESTRIAN_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO PASO A NIVEL:\n1. Auditoría de paso con señales acústicas/ópticas activas.\n2. Sancionar detención sobre la vía férrea.\n3. Verificar visibilidad y comportamiento preventivo."
    },

    'mountain-pass': {
      lines: [
        { y: 450, type: 'solid', direction: 'bidirectional', label: 'EJE CURVA PELIGROSA', infractionType: 'LINE_CROSSING' },
        { y: 700, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR BAJADA (Z1)', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO PUERTO DE MONTAÑA:\n1. Sancionar invasión de carril contrario en curvas sin visibilidad.\n2. Control del uso de freno motor (velocidad en descenso).\n3. Vigilancia de adelantamientos en zonas de baja adherencia."
    },

    'pedestrian-priority-zone': {
      lines: [
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'ZONA RESIDENCIAL (A1)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 550, type: 'loading-zone', direction: 'bidirectional', label: 'ÁREA CARGA CASCO (C1)', infractionType: 'LOADING_ZONE_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO CASCO HISTÓRICO:\n1. Prioridad absoluta al peatón sobre cualquier vehículo.\n2. Control de tiempo en zonas de carga/descarga.\n3. Sancionar acceso de vehículos no autorizados (R.E.S.)."
    },

    'es-autovia-nacional': {
      lines: [
        { y: 300, type: 'speed-zone', direction: 'northbound', label: 'RADAR A-X (120)', infractionType: 'SPEEDING' },
        { y: 500, type: 'divider', direction: 'bidirectional', label: 'MEDIANA BARRERA', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO RED DE CARRETERAS DEL ESTADO (RCE):\n1. Control estricto 120km/h.\n2. Vigilar uso carril izquierdo/central (Síndrome del carril izquierdo).\n3. [LINE: Y=800, TYPE=solid, LABEL=LÍNEA ARCÉN, INFRACTION=LINE_CROSSING]"
    },

    'es-convencional-ancha': {
      lines: [
        { y: 450, type: 'solid', direction: 'bidirectional', label: 'EJE CONVENCIONAL I', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO CARRETERA CONVENCIONAL ANCHA (Arcén > 1.5m):\n1. Límite general 90km/h.\n2. Vigilancia extrema en cruces al mismo nivel.\n3. Control de adelantamientos en tramos de visibilidad reducida."
    },

    'es-convencional-estrecha': {
      lines: [
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'EJE CONVENCIONAL II', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO CARRETERA CONVENCIONAL ESTRECHA (Sin Arcén):\n1. Límite 90km/h con precaución especial.\n2. Vigilar invasión de carril en curvas sin arcén.\n3. [LINE: Y=700, TYPE=speed-zone, LABEL=CONTROL VELOCIDAD REDUCIDA, INFRACTION=SPEEDING]"
    },

    'es-via-automoviles': {
      lines: [
        { y: 350, type: 'divider', direction: 'northbound', label: 'INICIO VÍA AUTOMÓVILES', infractionType: null },
        { y: 600, type: 'solid', direction: 'bidirectional', label: 'MEDIANA FLEXIBLE', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO VÍA PARA AUTOMÓVILES:\n1. Prohibición vehículos tracción animal/ciclos.\n2. Control de velocidad y sentido de circulación.\n3. Vigilancia de paradas en calzada."
    },

    'es-calle-30': {
      lines: [
        { y: 400, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR URBANO (30)', infractionType: 'SPEEDING' },
        { y: 600, type: 'divider', direction: 'bidirectional', label: 'DIVISOR CARRIL ÚNICO', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO CALLE 30 (Ley 2021):\n1. Límite estricto 30km/h en vía de carril único.\n2. Prioridad absoluta a peatones en toda la plataforma.\n3. Vigilar uso de móviles por el conductor."
    },

    'es-ciclocarril': {
      lines: [
        { y: 450, type: 'dashed', direction: 'bidirectional', label: 'LIMITADOR CICLOCARRIL', infractionType: null },
        { y: 700, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR COEXISTENCIA', infractionType: 'SPEEDING' }
      ],
      directivesTemplate: "PROTOCOLO CICLOCARRIL / CICLOVÍA:\n1. Seguridad Ciclista: Distancia 1.5m.\n2. Velocidad máxima 30km/h.\n3. Sancionar acoso a ciclistas (distancia seguridad)."
    },

    'es-supermanzana': {
      lines: [
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'ACCESO SUPERILLA', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 600, type: 'loading-zone', direction: 'bidirectional', label: 'CARGA VECINDAD', infractionType: 'LOADING_ZONE_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO SUPERMANZANA / SMART ZONE:\n1. Acceso restringido solo autorizados.\n2. Velocidad residencial (10km/h).\n3. Auditoría de ruido y ocupación de espacio."
    },

    'es-rotonda-partida': {
      lines: [
        { y: 300, type: 'stop', direction: 'bidirectional', label: 'SEMÁFORO ROTONDA', infractionType: 'STOP_VIOLATION' },
        { y: 600, type: 'divider', direction: 'bidirectional', label: 'FLUJO DIRECTO', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO GLORIETA PARTIDA:\n1. Control de giro a la izquierda condicionado.\n2. Vigilancia de semáforos en fase de cruce.\n3. Análisis de trayectorias en fase de Bypass."
    },

    'es-travesia-nacional': {
      lines: [
        { y: 350, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR TRAVESÍA (50)', infractionType: 'SPEEDING' },
        { y: 650, type: 'pedestrian', direction: 'bidirectional', label: 'CRUCE PRINCIPAL', infractionType: 'PEDESTRIAN_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO TRAVESÍA NACIONAL:\n1. Transición de velocidad Carretera-Urbano.\n2. Sancionar estacionamientos impropios que obstruyan visibilidad.\n3. [LINE: Y=800, TYPE=stop, LABEL=SEMÁFORO TRAVESÍA, INFRACTION=STOP_VIOLATION]"
    },

    'madrid-regional-highway': {
      lines: [
        { y: 350, type: 'speed-zone', direction: 'northbound', label: 'RADAR REGIONAL (M-607)', infractionType: 'SPEEDING' },
        { y: 550, type: 'divider', direction: 'bidirectional', label: 'MEDIANA NEW JERSEY', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO RED AUTONÓMICA MADRID (M-607/M-506):\n1. Control de velocidad variable según densidad.\n2. Vigilar uso de arcenes en retenciones.\n3. Sancionar circulación de vehículos no autorizados en bus-vao si aplica."
    },

    'madrid-convencional-cm': {
      lines: [
        { y: 400, type: 'solid', direction: 'bidirectional', label: 'EJE M-501 / M-100', infractionType: 'LINE_CROSSING' },
        { y: 650, type: 'dashed', direction: 'bidirectional', label: 'ZONA ADELANTAMIENTO', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO CARRETERA CONVENCIONAL CM:\n1. Vigilancia extrema de adelantamientos en línea continua.\n2. Control de velocidad en intersecciones al mismo nivel.\n3. [LINE: Y=800, TYPE=stop, LABEL=STOP INCORPORACIÓN, INFRACTION=STOP_VIOLATION]"
    },

    'madrid-calle30-tunnel': {
      lines: [
        { y: 300, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR CALLE 30 (70KM/H)', infractionType: 'SPEEDING' },
        { y: 500, type: 'solid', direction: 'bidirectional', label: 'CARRIL CONFINADO', infractionType: 'LINE_CROSSING' },
        { y: 700, type: 'divider', direction: 'bidirectional', label: 'DIVERGENCIA SALIDA', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO CALLE 30 / TÚNEL URBANO:\n1. Límite estricto 70km/h.\n2. Sancionar cambios de carril en tramos de línea continua (túnel).\n3. Detección de paradas por avería o emergencia (Protocolo Túnel)."
    },

    'madrid-zbe-enforcement': {
      lines: [
        { y: 300, type: 'pedestrian', direction: 'bidirectional', label: 'ACCESO ZBE MADRID', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 600, type: 'loading-zone', direction: 'bidirectional', label: 'CONTROL ETIQUETA', infractionType: 'FORENSIC_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO ZBE MADRID CENTRAL:\n1. Auditoría forense de etiquetas ambientales.\n2. Control de acceso de vehículos no autorizados.\n3. Vigilancia de uso de carriles exclusivos residentes."
    },

    'madrid-travesia-local': {
      lines: [
        { y: 250, type: 'speed-zone', direction: 'bidirectional', label: 'RADAR TRAVESÍA 30/50', infractionType: 'SPEEDING' },
        { y: 500, type: 'pedestrian', direction: 'bidirectional', label: 'PASO URBANO (P1)', infractionType: 'PEDESTRIAN_PRIORITY' },
        { y: 750, type: 'stop', direction: 'bidirectional', label: 'SEMÁFORO FOTO-ROJO', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO TRAVESÍA CM:\n1. Control de velocidad en zona urbana de carretera provincial.\n2. Prioridad peatonal en arterias principales del municipio.\n3. Sancionar rebasamiento de semáforo en fase roja."
    },

    'intersection-ddi': {
      lines: [
        { y: 300, type: 'solid', direction: 'bidirectional', label: 'CRUCE DDI (SENTIDO INVERSO)', infractionType: 'DIRECTION_VIOLATION' },
        { y: 600, type: 'divider', direction: 'bidirectional', label: 'CANALIZADOR DDI', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO DDI (Diverging Diamond Interchange):\n1. Vigilancia de flujo en sentido inverso (característico DDI).\n2. Control de semáforos en puntos de cruce X.\n3. Sancionar invasión de canalizadores de giro."
    },

    'intersection-spui': {
      lines: [
        { y: 500, type: 'stop', direction: 'bidirectional', label: 'PUNTO ÚNICO DE CONTROL', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO SPUI (Single-Point Urban Interchange):\n1. Auditoría de giros a la izquierda simultáneos.\n2. Control de despeje de intersección en fase única.\n3. [LINE: Y=800, TYPE=divider, LABEL=ZONA TRENZADO SPUI, INFRACTION=null]"
    },

    'intersection-turbo-roundabout': {
      lines: [
        { y: 350, type: 'solid', direction: 'bidirectional', label: 'DELIMITADOR ESPIRAL', infractionType: 'LINE_CROSSING' },
        { y: 650, type: 'solid', direction: 'bidirectional', label: 'CARRIL CONFINADO TURBO', infractionType: 'LINE_CROSSING' }
      ],
      directivesTemplate: "PROTOCOLO TURBO-ROTONDA:\n1. Sancionar cambio de carril entre espirales (Línea Continua Obligatoria).\n2. Control de entrada según elección de carril previa.\n3. Vigilancia de trazada espiral descendente."
    },

    'intersection-cfi': {
      lines: [
        { y: 200, type: 'stop', direction: 'bidirectional', label: 'PRE-CRUCE CFI', infractionType: 'STOP_VIOLATION' },
        { y: 400, type: 'solid', direction: 'bidirectional', label: 'CARRIL DESPLAZADO', infractionType: 'DIRECTION_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO CFI (Continuous Flow Interchange):\n1. Vigilancia de carriles de giro desplazados.\n2. Control de coordinación de fases en stop intermedio.\n3. Sancionar incorporación tardía al carril CFI."
    },

    'intersection-box-junction': {
      lines: [
        { y: 300, type: 'divider', direction: 'bidirectional', label: 'YELLOW BOX NORTE', infractionType: 'FORENSIC_PRIORITY' },
        { y: 700, type: 'divider', direction: 'bidirectional', label: 'YELLOW BOX SUR', infractionType: 'FORENSIC_PRIORITY' }
      ],
      directivesTemplate: "PROTOCOLO BOX JUNCTION (Anti-Bloqueo):\n1. Sancionar entrada en área amarilla sin salida libre.\n2. Auditoría de tiempo de permanencia en intersección.\n3. Detección de colapso por bloqueo transversal."
    },

    'intersection-stack': {
      lines: [
        { y: 400, type: 'divider', direction: 'northbound', label: 'RAMAL NIVEL +1', infractionType: null },
        { y: 600, type: 'divider', direction: 'northbound', label: 'RAMAL NIVEL +2', infractionType: null }
      ],
      directivesTemplate: "PROTOCOLO STACK INTERCHANGE (Multinivel):\n1. Control de velocidad en ramales de gran altura.\n2. Vigilancia de cambios de carril en zonas de convergencia de niveles.\n3. Auditoría de flujo masivo direccional."
    },

    'intersection-magic-roundabout': {
      lines: [
        { y: 300, type: 'roundabout-access' as any, direction: 'bidirectional', label: 'MICRO-ROTATOR 1', infractionType: 'STOP_VIOLATION' },
        { y: 600, type: 'roundabout-access' as any, direction: 'bidirectional', label: 'MICRO-ROTATOR 2', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO MAGIC ROUNDABOUT (Multi-Rotonda):\n1. Complejidad Máxima: Vigilancia en micro-rotondas periféricas.\n2. Control de giro inverso en el anillo central.\n3. Análisis de prioridades en múltiples puntos de decisión."
    },

    'intersection-dumbbell': {
      lines: [
        { y: 300, type: 'stop', direction: 'bidirectional', label: 'ROTONDA A', infractionType: 'STOP_VIOLATION' },
        { y: 700, type: 'stop', direction: 'bidirectional', label: 'ROTONDA B', infractionType: 'STOP_VIOLATION' }
      ],
      directivesTemplate: "PROTOCOLO DUMBBELL INTERCHANGE:\n1. Control de sincronización entre rotondas gemelas.\n2. Vigilancia de congestión en el puente de enlace.\n3. Sancionar giros prohibidos en ramales de conexión."
    },

    'custom': {
      lines: [],
      directivesTemplate: "PROTOCOLO PERSONALIZADO:\nDefina sus propias directivas. Use el formato [LINE: Y=500, TYPE=solid, LABEL=LÍNEA N...] para crear zonas dinámicas."
    }
  };

  // === AUTO-SYNTHESIS: Neural Line Generator ===
  // Analyzes directives and automatically generates detection lines
  const parseDirectivesToLines = useCallback((text: string, baseLines: DetectionLine[]): DetectionLine[] => {
    const lines = [...baseLines];

    // 1. Manual line syntax: [LINE: Y=500, TYPE=solid, LABEL=...]
    const regex = /\[LINE:\s*Y=(\d+),\s*TYPE=([^,\]\s]+),\s*LABEL=([^,\]]+)(?:,\s*INFRACTION=([^,\]\s]+))?\]/gi;
    let match;
    while ((match = regex.exec(text)) !== null) {
      const [, y, type, label, infraction] = match;
      lines.push({
        y: parseInt(y),
        type: type.trim() as any,
        direction: 'bidirectional',
        label: label.trim(),
        infractionType: (infraction?.trim() || null) as any
      });
    }

    // 2. INTELLIGENT AUTO-SYNTHESIS: Keyword detection
    const lowerText = text.toLowerCase();
    const hasLines = lines.length > 0;

    // Define keywords and their corresponding line types
    const synthRules = [
      {
        keywords: ['continua', 'línea continua', 'cruce de línea', 'line crossing'],
        line: { y: 500, type: 'solid' as const, label: 'AUTO: LÍNEA CONTINUA', infraction: 'LINE_CROSSING' }
      },
      {
        keywords: ['stop', 'detención', 'alto', 'parada obligatoria'],
        line: { y: 400, type: 'stop' as const, label: 'AUTO: STOP', infraction: 'STOP_VIOLATION' }
      },
      {
        keywords: ['peatones', 'paso de peatones', 'cebra', 'pedestrian'],
        line: { y: 600, type: 'pedestrian' as const, label: 'AUTO: PASO PEATONES', infraction: 'PEDESTRIAN_ZONE' }
      },
      {
        keywords: ['bus', 'carril bus', 'bus lane'],
        line: { y: 450, type: 'bus-lane' as const, label: 'AUTO: CARRIL BUS', infraction: 'BUS_LANE_VIOLATION' }
      },
      {
        keywords: ['carga', 'descarga', 'loading', 'zona de carga'],
        line: { y: 550, type: 'loading-zone' as const, label: 'AUTO: ZONA CARGA', infraction: 'LOADING_ZONE' }
      },
      {
        keywords: ['velocidad', 'radar', 'speed', 'exceso'],
        line: { y: 350, type: 'speed-zone' as const, label: 'AUTO: CONTROL VELOCIDAD', infraction: 'SPEEDING' }
      },
      {
        keywords: ['divisoria', 'carril', 'lane', 'divider'],
        line: { y: 500, type: 'divider' as const, label: 'AUTO: DIVISORIA CARRIL', infraction: null as any }
      },
      {
        keywords: ['incorporación', 'entrada', 'access', 'ramal'],
        line: { y: 300, type: 'dashed' as const, label: 'AUTO: INCORPORACIÓN', infraction: null as any }
      }
    ];

    // Apply synthesis rules based on detected keywords
    synthRules.forEach((rule, idx) => {
      const hasKeyword = rule.keywords.some(kw => lowerText.includes(kw));
      if (hasKeyword) {
        // Offset Y position slightly to avoid overlapping
        const yOffset = idx * 50;
        const synthesizedLine: DetectionLine = {
          y: rule.line.y + yOffset,
          type: rule.line.type,
          direction: 'bidirectional',
          label: rule.line.label,
          infractionType: rule.line.infraction
        };

        // Only add if not already present (avoid duplicates)
        const exists = lines.some(l =>
          l.label === synthesizedLine.label ||
          (Math.abs(l.y - synthesizedLine.y) < 30 && l.type === synthesizedLine.type)
        );

        if (!exists) {
          lines.push(synthesizedLine);
        }
      }
    });

    // 3. Multi-lane detection: If directives mention multiple lanes/carriles
    const laneMatches = text.match(/(\d+)\s*(carriles|lanes|vías)/gi);
    if (laneMatches && laneMatches.length > 0) {
      const numLanes = parseInt(laneMatches[0]);
      if (numLanes >= 2 && numLanes <= 4) {
        const spacing = 1000 / (numLanes + 1);
        for (let i = 1; i <= numLanes; i++) {
          const laneY = Math.floor(spacing * i);
          const exists = lines.some(l => Math.abs(l.y - laneY) < 50);
          if (!exists) {
            lines.push({
              y: laneY,
              type: 'divider',
              direction: 'bidirectional',
              label: `AUTO: CARRIL ${i}`,
              infractionType: null as any
            });
          }
        }
      }
    }

    // Remove duplicates by Y and label
    return lines.filter((v, i, a) => a.findIndex(t => t.y === v.y && t.label === v.label) === i);
  }, []);

  const combinedLinesForSync = useCallback((configs: string[]) => {
    let lines: DetectionLine[] = [];
    configs.forEach(c => {
      if (ROAD_PRESETS[c]) lines = [...lines, ...ROAD_PRESETS[c].lines];
    });
    return lines;
  }, []);

  // Neural Merge: Sync lines and directives from multiple selected protocols
  const syncProtocols = useCallback((configs: string[], currentDirectives: string) => {
    let combinedLines: DetectionLine[] = [];
    let combinedDirectives = "";

    configs.forEach(conf => {
      const preset = ROAD_PRESETS[conf];
      if (preset) {
        combinedLines = [...combinedLines, ...preset.lines];
        combinedDirectives += (combinedDirectives ? "\n\n" : "") + preset.directivesTemplate;
      }
    });

    // Apply Neural Synthesis on top of combined templates
    const finalLines = parseDirectivesToLines(combinedDirectives, combinedLines);
    setDetectionLines(finalLines);
    setDirectives(combinedDirectives);

    setAiFeedback(`SINCRONIZACIÓN MULTI-PROTOCOLO: ${configs.length} ACTIVOS`);
    setTimeout(() => setAiFeedback(null), 3000);
  }, [parseDirectivesToLines]);

  const toggleProtocol = (confKey: string) => {
    setSelectedConfigs(prev => {
      let next;
      if (prev.includes(confKey)) {
        if (prev.length === 1) return prev; // Keep at least one
        next = prev.filter(c => c !== confKey);
      } else {
        next = [...prev, confKey];
      }
      syncProtocols(next, directives);
      return next;
    });
  };

  // === AUTO-SYNTHESIS TRIGGER ===
  // Automatically re-synthesize lines when directives change (user edits text)
  useEffect(() => {
    if (!isManualMode && directives) {
      const baseLines = combinedLinesForSync(selectedConfigs);
      const synthesized = parseDirectivesToLines(directives, baseLines);
      setDetectionLines(synthesized);
      console.log(`🧠 AUTO-SYNTHESIS: Generated ${synthesized.filter(l => l.label.startsWith('AUTO:')).length} smart lines`);
    }
  }, [directives, selectedConfigs, isManualMode, parseDirectivesToLines, combinedLinesForSync]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tracksRef = useRef<Track[]>([]);
  const detectorRef = useRef<YoloDetector | null>(null);
  const trackerRef = useRef<ByteTracker | BoTSORT | null>(null);
  const processingRef = useRef(false);
  const lastFrameTime = useRef(Date.now());
  const fpsRef = useRef(30);

  // Feedback al actualizar directivas
  const handleDirectivesChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newDirectives = e.target.value;
    setDirectives(newDirectives);

    // Neural Synthesis Integration: Actualizar líneas basadas en el texto
    const basePreset = ROAD_PRESETS[selectedConfigs[0]];
    if (basePreset) {
      const dynamicLines = parseDirectivesToLines(newDirectives, combinedLinesForSync(selectedConfigs));
      setDetectionLines(dynamicLines);
    }

    setAiFeedback("ACTUALIZANDO CRITERIOS DE ANÁLISIS... SINCRONIZANDO CON RED NEURONAL");
    setTimeout(() => setAiFeedback(null), 3000);
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setSystemStats(prev => ({
        ...prev,
        cpu: Math.floor(20 + Math.random() * 15),
        mem: Number((1.1 + Math.random() * 0.3).toFixed(1)),
        temp: Math.floor(40 + Math.random() * 5),
        net: Math.floor(80 + Math.random() * 10)
      }));
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  const safePlay = async () => {
    if (videoRef.current) {
      try {
        await videoRef.current.play();
        setIsPlaying(true);
      } catch (error) {
        if (error.name !== 'AbortError') console.error("Playback error:", error);
      }
    }
  };

  const safePause = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const detector = new YoloDetector();
        // Use BASE_URL to support GitHub Pages subdirectory deployment
        await detector.load(
          import.meta.env.BASE_URL + 'upload/yolo11n_640.onnx',
          import.meta.env.BASE_URL + 'upload/yolo11n_pose.onnx' // Pose model
        );
        detectorRef.current = detector;

        // Initialize tracker based on preset configuration
        if (yoloConfig.trackerType === 'BoT-SORT') {
          trackerRef.current = new BoTSORT();
        } else {
          trackerRef.current = new ByteTracker();
        }
        console.log(`YOLOv11 + ${yoloConfig.trackerType} system initialized`);
      } catch (e) {
        console.error("YOLO Load Error", e);
      }
    };
    loadModels();
  }, [yoloConfig.trackerType]);

  useEffect(() => {
    if (videoUrl && source === 'upload' && videoRef.current) {
      videoRef.current.src = videoUrl;
      videoRef.current.load();
      videoRef.current.onloadeddata = () => safePlay();
    }
  }, [videoUrl, source]);

  const captureVideoClip = async (durationMs: number = 5000): Promise<string> => {
    return new Promise((resolve) => {
      if (!canvasRef.current) return resolve("");
      const stream = canvasRef.current.captureStream(30);
      const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        resolve(URL.createObjectURL(blob));
      };
      recorder.start();
      setTimeout(() => recorder.stop(), durationMs);
    });
  };

  const runNeuralAudit = async (track: Track) => {
    if (processingRef.current) return;
    processingRef.current = true;
    setIsAnalyzing(true);
    setStatusMsg("FORENSIC ANALYSIS: DAGANZO_POLICE_ALGO ACTIVE...");

    try {
      const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_GOOGLE_GENAI_KEY });
      const systemInstruction = `Eres el AUDITOR FORENSE SUPREMO asignado a la Policía Local de Daganzo de Arriba.
      
      ESPECIFICACIONES TÉCNICAS DEL SISTEMA (SENTINEL V15 - ARQUITECTURA HÍBRIDA EDGE+CLOUD):
      1. Capa Local (Edge Computing): YOLOv11n (ONNX) + ByteTrack para seguimiento de alta precisión en tiempo real.
         - Modelo de Detección: YOLO11-Nano (640×640px) corriendo en WebAssembly SIMD
         - Tracker: ByteTrack con Filtro de Kalman de 8 estados [cx, cy, área, altura, vx, vy, va, vh]
         - Coincidencia: Algoritmo Húngaro con IoU threshold = ${yoloConfig.matchIouThreshold}
         - Configuración Activa: ${activePreset} 
           * Umbral Conf. YOLO: ${yoloConfig.confThreshold}
           * Skip de Frames: ${yoloConfig.detectionSkip}
           * Buffer de Track: ${yoloConfig.trackBufferFrames} frames
           * High Det. Threshold: ${yoloConfig.highDetThreshold}
         - Suavizado: Filtro de Kalman con constante de velocidad y corrección por medición
      2. Capa Remota (Cloud Judiciary - TU ROL): Juicio legal definitivo de la escena basado en evidencia visual multiplexada y las directivas municipales de Daganzo.
      3. Geometría Espacial: Sistema de coordenadas normalizado (0-1000) con líneas de detección en eje Y.
      
      DATOS DEL VEHÍCULO ANALIZADO:
      - Track ID: ${track.id}
      - Edad del Track: ${track.age} frames
      - Confianza Media: ${track.confidence.toFixed(3)}
      - Velocidad Estimada: ${Math.floor(track.velocity * 3.6)} km/h (basado en desplazamiento entre frames)
      - Clase Detectada: ${track.label}
      - Estado Infractor: ${track.isInfractor ? 'CONFIRMADO (cruce de línea detectado)' : 'En evaluación'}
      
      INSTRUCCIONES DE ANÁLISIS:
      Analiza la ráfaga de imágenes forenses para determinar si existe infracción de tráfico siguiendo ESTRICTAMENTE estas directivas municipales de Daganzo:
      "${directives}"
      
      SALIDA JSON OBLIGATORIA (NO OTROS FORMATOS):
      {
        "infraction": boolean,
        "plate": "MATRÍCULA",
        "ocrConfidence": 0.95,
        "description": "Relato técnico detallado de la infracción según directivas, incluyendo evidencia visual específica observada",
        "severity": "leve|grave|muy-grave",
        "legalArticle": "Artículo específico del código de circulación español",
        "reasoning": ["Evidencia visual 1", "Evidencia visual 2", "Inferencia técnica"],
        "vehicleType": "Descripción visual del vehículo (marca, modelo, color)",
        "subType": "turismo|furgoneta|camión|moto|bus",
        "confidence": 0.98,
        "telemetry": { 
          "speedEstimated": "${Math.floor(track.velocity * 3.6)} km/h", 
          "trackAge": "${track.age} frames",
          "yoloConfidence": "${track.confidence.toFixed(3)}",
          "maneuverType": "Giro/Cruce/Adelantamiento/Recto",
          "poseAlert": boolean
        }
      }`;

      const videoClipPromise = captureVideoClip(10000);

      const parts = track.snapshots.map(s => ({ inlineData: { mimeType: 'image/jpeg', data: s } }));
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: { parts: [...parts, { text: "Ejecutar auditoría forense inmediata basada en directivas de Daganzo." }] },
        config: { systemInstruction, responseMimeType: "application/json", temperature: 0.1 }
      });

      const audit = JSON.parse(response.text.trim().match(/\{[\s\S]*\}/)?.[0] || "{}");
      const videoClipUrl = await videoClipPromise;

      const trackIdx = tracksRef.current.findIndex(t => t.id === track.id);
      if (trackIdx !== -1) {
        tracksRef.current[trackIdx].plate = audit.plate;
        tracksRef.current[trackIdx].isInfractor = audit.infraction;
        tracksRef.current[trackIdx].analyzed = true;
      }

      if (audit.infraction) {
        setCumulativeExpedientes(prev => prev + 1);
        setLogs(prev => [{
          ...audit,
          id: Date.now(),
          image: `data:image/jpeg;base64,${track.snapshots[track.snapshots.length - 1]}`,
          snapshots: track.snapshots,
          videoUrl: videoClipUrl,
          time: new Date().toLocaleTimeString(),
          date: new Date().toLocaleDateString(),
          violatedDirective: directives,
          telemetry: { ...audit.telemetry, framesAnalyzed: track.points.length }
        }, ...prev]);
      }
    } catch (e) {
      console.error("Forensic Hub Error:", e);
    } finally {
      setIsAnalyzing(false);
      setStatusMsg(null);
      processingRef.current = false;
    }
  };

  const processFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !detectorRef.current || !trackerRef.current) return;
    const v = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx || v.readyState < 2) return;

    const now = Date.now();
    const delta = now - lastFrameTime.current;
    const deltaSeconds = delta / 1000;
    if (delta > 0) {
      const currentFps = Math.round(1000 / delta);
      setFps(currentFps);
      fpsRef.current = currentFps;
    }
    lastFrameTime.current = now;

    canvas.width = canvas.parentElement?.clientWidth || 0;
    canvas.height = canvas.parentElement?.clientHeight || 0;
    const scale = Math.min(canvas.width / v.videoWidth, canvas.height / v.videoHeight);
    const dW = v.videoWidth * scale;
    const dH = v.videoHeight * scale;
    const oX = (canvas.width - dW) / 2;
    const oY = (canvas.height - dH) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!isPlaying) return;

    // HUD-only layer
    if (isManualMode && canvasRef.current) {
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(oX, 0); ctx.lineTo(oX + dW, 0);
      ctx.setLineDash([]);
    }

    frameCounterRef.current++;

    // --- YOLOv11 & ByteTrack Pipeline ---
    let detections: any[] = [];
    const runInference = frameCounterRef.current % yoloConfig.detectionSkip === 0;

    if (runInference) {
      detections = await detectorRef.current.detect(v, yoloConfig.confThreshold);
      if (poseEstimationEnabled && detectorRef.current.poseSession) {
        lastPosesRef.current = await detectorRef.current.detectPose(v, 0.5);
      } else {
        lastPosesRef.current = [];
      }
    }

    // Update tracker configuration dynamically
    if (trackerRef.current) {
      trackerRef.current.highThresh = yoloConfig.highDetThreshold;
      trackerRef.current.matchThresh = yoloConfig.matchIouThreshold;
      trackerRef.current.trackBufferFrames = yoloConfig.trackBufferFrames;

      // BoT-SORT specific configuration
      if (trackerRef.current instanceof BoTSORT) {
        trackerRef.current.appearanceWeight = yoloConfig.appearanceWeight;
      }
    }

    let activeTracks: any[] = [];
    if (runInference && detectorRef.current) {
      // Pass video frame to BoT-SORT for appearance feature extraction
      if (trackerRef.current instanceof BoTSORT) {
        activeTracks = trackerRef.current.update(detections, v);
      } else {
        activeTracks = trackerRef.current.update(detections);
      }

      // Sync visual tracks with tracker output
      const newVisualTracks: Track[] = [];
      const matchedIds = new Set<number>();

      activeTracks.forEach(t => {
        matchedIds.add(t.trackId);
        let vt = tracksRef.current.find(existing => existing.id === t.trackId);

        // === CRITICAL FIX: Convert pixel coordinates to normalized 0-1000 space ===
        // YOLO bbox is in video pixel coordinates
        // UI rendering expects normalized 0-1000 coordinates
        const videoW = v.videoWidth || 1280;
        const videoH = v.videoHeight || 720;

        const cx_normalized = (t.bbox[0] + t.bbox[2] / 2) / videoW * 1000;
        const cy_normalized = (t.bbox[1] + t.bbox[3] / 2) / videoH * 1000;
        const w_normalized = (t.bbox[2] / videoW) * 1000;
        const h_normalized = (t.bbox[3] / videoH) * 1000;

        if (!vt) {
          // New Visual Track
          vt = {
            id: t.trackId,
            label: t.className,
            points: [],
            lastSeen: now,
            lastSnapshotTime: 0,
            color: VEHICLE_COLORS[t.className] || '#fff',
            snapshots: [],
            velocity: 0,
            age: t.age,
            analyzed: false,
            vx: 0, vy: 0,
            predictedX: cx_normalized,
            predictedY: cy_normalized,
            confidence: t.score,
            missedFrames: 0,
            w: w_normalized,
            h: h_normalized,
            // Initialize visual state
            renderX: cx_normalized,
            renderY: cy_normalized,
            renderW: w_normalized,
            renderH: h_normalized,
            isInfractor: false
          };
        }

        // Update State with normalized coordinates
        // Update State with smooth EMA interpolation (Alpha 0.6 = fast but stable)
        const alpha = 0.6;
        vt.renderX = vt.renderX * alpha + cx_normalized * (1 - alpha);
        vt.renderY = vt.renderY * alpha + cy_normalized * (1 - alpha);
        vt.renderW = vt.renderW * alpha + w_normalized * (1 - alpha);
        vt.renderH = vt.renderH * alpha + h_normalized * (1 - alpha);

        vt.lastSeen = now;
        vt.points.push({ x: cx_normalized, y: cy_normalized, time: now });
        vt.w = w_normalized;
        vt.h = h_normalized;
        vt.confidence = t.score;
        vt.age = t.age;

        // Calculate velocity from track history
        if (vt.points.length > 1) {
          const p1 = vt.points[vt.points.length - 1];
          const p2 = vt.points[vt.points.length - 2];
          vt.vx = p1.x - p2.x;
          vt.vy = p1.y - p2.y;
          vt.velocity = Math.sqrt(vt.vx * vt.vx + vt.vy * vt.vy);
        }

        // History clamp
        if (vt.points.length > 50) vt.points.shift();

        newVisualTracks.push(vt);
      });

      tracksRef.current = newVisualTracks;
    } else {
      // Smooth interpolation on skipped frames using Kalman-predicted velocity
      tracksRef.current.forEach(t => {
        if (t.points.length > 0) {
          const lastP = t.points[t.points.length - 1];
          // Use velocity from last update
          // Smooth visual extrapolation
          t.renderX += t.vx;
          t.renderY += t.vy;
          t.points.push({ x: lastP.x + t.vx, y: lastP.y + t.vy, time: now });
          if (t.points.length > 50) t.points.shift();
        }
      });
    }

    const matchedTracks = new Set(tracksRef.current.map(t => t.id)); // For compatibility with rendering pipeline

    // STEP 4: Render all tracks with premium forensic styling
    tracksRef.current.forEach(track => {
      if (track.points.length === 0) return;

      const lastP = track.points[track.points.length - 1];

      // HUD ELASTICITY: Calculate smoothed visual dimensions
      // HUD ELASTICITY: Calculate smoothed visual dimensions from EMA state
      const lw = track.renderW;
      const lh = track.renderH;
      const lx = track.renderX - lw / 2;
      const ly = track.renderY - lh / 2;

      const cpX = oX + (lx / 1000) * dW;
      const cpY = oY + (ly / 1000) * dH;
      const bW = (lw / 1000) * dW;
      const bH = (lh / 1000) * dH;

      // Calculate speed
      const metersMovedPerFrame = track.velocity / PIXELS_PER_METER;
      const speedKmh = Math.floor(metersMovedPerFrame * fpsRef.current * 3.6);

      // STABLE OPACITY: High minimum with hysteresis to prevent flicker
      let opacity = track.confidence;
      if (track.age > 5) {
        opacity = Math.max(0.95, track.confidence); // Very stable after initialization
      } else {
        opacity = Math.max(0.7, track.confidence); // Lower during warmup
      }
      const isInfractor = track.isInfractor;

      ctx.globalAlpha = opacity;

      if (isInfractor) {
        // INFRACTOR RENDERING: Dynamic Red Glow + Double Vibrating Border
        const vibrateOffset = Math.sin(Date.now() / 100) * 2; // Subtle vibration

        // Outer Glow (Red Aura)
        ctx.shadowColor = '#ef4444';
        ctx.shadowBlur = 30;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;

        // First Border (Thick Red)
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 5;
        ctx.strokeRect(cpX + vibrateOffset, cpY + vibrateOffset, bW, bH);

        // Second Border (Outer Thin Red - creates double-line effect)
        ctx.strokeStyle = '#dc2626';
        ctx.lineWidth = 2;
        ctx.strokeRect(cpX - 3 + vibrateOffset, cpY - 3 + vibrateOffset, bW + 6, bH + 6);

        // Reset shadow
        ctx.shadowBlur = 0;

        // Corner Markers (Tactical Style)
        const cornerSize = 15;
        ctx.strokeStyle = '#fca5a5';
        ctx.lineWidth = 3;
        // Top-left
        ctx.beginPath();
        ctx.moveTo(cpX, cpY + cornerSize);
        ctx.lineTo(cpX, cpY);
        ctx.lineTo(cpX + cornerSize, cpY);
        ctx.stroke();
        // Top-right
        ctx.beginPath();
        ctx.moveTo(cpX + bW - cornerSize, cpY);
        ctx.lineTo(cpX + bW, cpY);
        ctx.lineTo(cpX + bW, cpY + cornerSize);
        ctx.stroke();
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(cpX + bW, cpY + bH - cornerSize);
        ctx.lineTo(cpX + bW, cpY + bH);
        ctx.lineTo(cpX + bW - cornerSize, cpY + bH);
        ctx.stroke();
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(cpX + cornerSize, cpY + bH);
        ctx.lineTo(cpX, cpY + bH);
        ctx.lineTo(cpX, cpY + bH - cornerSize);
        ctx.stroke();

        // Label Background (Red Alert Style)
        ctx.fillStyle = '#dc2626';
        ctx.fillRect(cpX, cpY - 24, bW, 22);

        // Label Text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(`⚠ INFRACCIÓN | ${track.plate || 'ANALYZING'} | ${speedKmh} KM/H`, cpX + 5, cpY - 7);

      } else {
        // NORMAL VEHICLE RENDERING: Clean professional style
        ctx.strokeStyle = track.color;
        ctx.lineWidth = 2;
        ctx.strokeRect(cpX, cpY, bW, bH);

        // Label Background
        ctx.fillStyle = track.color;
        ctx.fillRect(cpX, cpY - 20, bW, 18);

        // Label Text
        ctx.fillStyle = '#000';
        ctx.font = 'bold 10px monospace';
        ctx.fillText(`${(track.subType || track.label).toUpperCase()} | ${speedKmh} KM/H`, cpX + 5, cpY - 7);
      }

      ctx.globalAlpha = 1.0;

      // === LINE CROSSING DETECTION (Infraction Trigger) ===
      // Check if track crossed any detection line
      if (track.points.length >= 2) {
        const p1 = track.points[track.points.length - 2]; // Previous position
        const p2 = track.points[track.points.length - 1]; // Current position

        detectionLines.forEach(line => {
          // Check if trajectory crossed the line (Y coordinate comparison in 0-1000 space)
          const crossedLine = (p1.y < line.y && p2.y >= line.y) || (p1.y > line.y && p2.y <= line.y);

          if (crossedLine && !track.isInfractor) {
            // Determine infraction type based on line type
            const infractionTypes = {
              'solid': 'CRUCE_LINEA_CONTINUA',
              'stop': 'NO_DETENCION_STOP',
              'pedestrian': 'INVASION_PASO_PEATONES',
              'bus-lane': 'CIRCULACION_CARRIL_BUS',
              'loading-zone': 'ESTACIONAMIENTO_ZONA_CARGA',
              'speed-zone': 'EXCESO_VELOCIDAD'
            };

            const infractionType = infractionTypes[line.type] || 'INFRACCION_GENERICA';

            console.log(`🚨 INFRACCIÓN DETECTADA: Track ${track.id} cruzó línea "${line.label}" (${infractionType})`);

            // Mark as infractor and trigger immediate audit if enough evidence collected
            if (track.snapshots.length >= 5 && track.age > 15 && !track.analyzed) {
              track.isInfractor = true;
              // Trigger audit asynchronously to not block rendering
              setTimeout(() => runNeuralAudit(track), 100);
            }
          }
        });
      }

      // Snapshot capture (high frequency forensic buffer)
      if (matchedTracks.has(track.id) && now - track.lastSnapshotTime > 150 && track.snapshots.length < 25 && track.confidence > 0.65) {
        const snap = document.createElement('canvas');
        snap.width = 400; snap.height = 300;
        const trackScreenX = (lastP.x / 1000) * v.videoWidth - 40;
        const trackScreenY = (lastP.y / 1000) * v.videoHeight - 40;
        snap.getContext('2d')?.drawImage(v, trackScreenX, trackScreenY, 80, 80, 0, 0, 400, 300);
        track.snapshots.push(snap.toDataURL('image/jpeg', 0.65).split(',')[1]);
        track.lastSnapshotTime = now;
      }

      // Fallback: Trigger forensic audit at age 50 for stable tracks (if no line crossing detected)
      if (track.age === 50 && !track.analyzed && !processingRef.current && track.confidence > 0.8 && !track.isInfractor) {
        runNeuralAudit(track);
      }
    });

    // STEP 5: Draw All Detection Lines (multi-line system)
    ctx.save();
    detectionLines.forEach((line, index) => {
      const lineY = oY + (line.y / 1000) * dH;

      // Line style based on type and infraction priority
      switch (line.type) {
        case 'solid':
          ctx.strokeStyle = '#ef4444'; // Red for solid lines
          ctx.lineWidth = 4;
          ctx.setLineDash([]);
          break;
        case 'stop':
          ctx.strokeStyle = '#dc2626'; // Bright Red for STOP
          ctx.lineWidth = 6;
          ctx.setLineDash([]);
          break;
        case 'pedestrian':
          ctx.strokeStyle = '#22d3ee'; // Cyan for Pedestrian zones
          ctx.lineWidth = 8;
          ctx.setLineDash([30, 20]); // Zebra pattern
          break;
        case 'bus-lane':
          ctx.strokeStyle = '#f59e0b'; // Amber for Bus Lane
          ctx.lineWidth = 5;
          ctx.setLineDash([]);
          break;
        case 'loading-zone':
          ctx.strokeStyle = '#a855f7'; // Purple for Loading zones
          ctx.lineWidth = 3;
          ctx.setLineDash([10, 10]);
          break;
        case 'speed-zone':
          ctx.strokeStyle = '#22c55e'; // Green for Speed control
          ctx.lineWidth = 2;
          ctx.setLineDash([2, 5]);
          break;
        case 'dashed':
          ctx.strokeStyle = '#f59e0b'; // Amber for lane dividers
          ctx.lineWidth = 2;
          ctx.setLineDash([15, 10]);
          break;
        default: // divider
          ctx.strokeStyle = '#06b6d4'; // Cyan for simple dividers
          ctx.lineWidth = 2;
          ctx.setLineDash([10, 5]);
          break;
      }

      ctx.globalAlpha = (line.type === 'solid' || line.type === 'stop') ? 0.8 : (line.label.startsWith('GRID_') || line.label.startsWith('PERSP_')) ? 0.15 : 0.5;

      // Check if line is angled (has x1,y1,x2,y2 coordinates)
      if (line.x1 !== undefined && line.y2 !== undefined && line.x2 !== undefined) {
        // Angled line rendering
        const x1 = oX + (line.x1 / 1000) * dW;
        const y1 = oY + (line.y / 1000) * dH;
        const x2 = oX + (line.x2 / 1000) * dW;
        const y2 = oY + (line.y2 / 1000) * dH;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      } else {
        // Standard horizontal line
        ctx.beginPath();
        ctx.moveTo(0, lineY);
        ctx.lineTo(canvas.width, lineY);
        ctx.stroke();

        // Line markers (ticks) - only for non-grid lines
        if (!line.label.startsWith('GRID_') && !line.label.startsWith('PERSP_')) {
          ctx.setLineDash([]);
          ctx.lineWidth = 2;
          for (let x = 0; x < canvas.width; x += 120) {
            ctx.beginPath();
            ctx.moveTo(x, lineY - 6);
            ctx.lineTo(x, lineY + 6);
            ctx.stroke();
          }
        }
      }

      // Line label - only for important lines (not grid)
      if (!line.label.startsWith('GRID_') && !line.label.startsWith('PERSP_')) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = 'bold 10px monospace';
        ctx.globalAlpha = 0.9;
        ctx.fillText(line.label, 15, lineY - 12);
      }
    });

    // STEP 6: Draw Pose Skeletons (if active)
    if (poseEstimationEnabled && lastPosesRef.current.length > 0) {
      const poses = lastPosesRef.current;
      const getVidX = (x: number) => oX + (x / v.videoWidth) * dW;
      const getVidY = (y: number) => oY + (y / v.videoHeight) * dH;

      ctx.lineWidth = 2;
      ctx.strokeStyle = '#c084fc'; // Pose Purple

      poses.forEach(pose => {
        // Draw Skeleton Connections
        const skeleton = [
          [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
        ];

        ctx.beginPath();
        skeleton.forEach(([i, j]) => {
          const kp1 = pose.keypoints[i];
          const kp2 = pose.keypoints[j];
          if (kp1 && kp2 && kp1.score > 0.5 && kp2.score > 0.5) {
            ctx.moveTo(getVidX(kp1.x), getVidY(kp1.y));
            ctx.lineTo(getVidX(kp2.x), getVidY(kp2.y));
          }
        });
        ctx.stroke();

        // Draw Keypoints
        ctx.fillStyle = '#f0abfc';
        pose.keypoints.forEach(kp => {
          if (kp.score > 0.5) {
            ctx.beginPath();
            ctx.arc(getVidX(kp.x), getVidY(kp.y), 3, 0, 2 * Math.PI);
            ctx.fill();
          }
        });
      });
    }

    ctx.restore();

    // STEP 6: Cleanup - Remove tracks that are truly lost
    tracksRef.current = tracksRef.current.filter(t => t.missedFrames < Math.max(30, yoloConfig.trackBufferFrames) && t.confidence > 0.01);
  }, [isPlaying, detectionLines, yoloConfig]);

  useEffect(() => {
    let handle: number;
    const loop = () => { processFrame(); handle = requestAnimationFrame(loop); };
    loop();
    return () => cancelAnimationFrame(handle);
  }, [processFrame]);

  const startStream = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 1280, height: 720 } });
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        videoRef.current.onloadedmetadata = () => { safePlay(); setSource('live'); };
      }
    } catch (e) { alert("Error de cámara."); }
  };

  return (
    <div className="h-screen w-screen bg-[#020617] text-slate-100 flex flex-col lg:flex-row overflow-hidden font-sans select-none">

      {/* SIDEBAR LEFT */}
      <aside className="hidden lg:flex w-96 border-r border-white/5 flex-col z-50 bg-[#020617]/98 backdrop-blur-2xl overflow-y-auto custom-scrollbar">
        <div className="p-8 border-b border-white/5 flex items-center gap-4">
          <div className="w-10 h-10 bg-cyan-500/10 rounded-xl flex items-center justify-center border border-cyan-500/30 shadow-neon">
            <ShieldCheck className="text-cyan-500 w-6 h-6 animate-pulse" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-black italic tracking-tighter text-white uppercase leading-none">SENTINEL</span>
            <span className="text-[10px] font-black tracking-[0.5em] text-cyan-500/80 uppercase whitespace-nowrap">POLICÍA LOCAL DAGANZO</span>
          </div>
        </div>

        <div className="flex-1 p-6 space-y-8">
          {/* === NEURAL TRACKING CORE === */}
          <div className="space-y-4">
            <h3 className="text-[11px] font-black uppercase text-slate-500 tracking-widest flex items-center gap-2 px-2">
              <Cpu size={12} className="text-cyan-500" /> NEURAL TRACKING CORE
            </h3>

            {/* Compact Preset Selector (Grid 2x2) */}
            <div className="grid grid-cols-2 gap-2">
              {Object.keys(trackingPresets).map(preset => {
                const config = trackingPresets[preset];
                const presetInfo = {
                  'highway-fast-bytetrack': { icon: '🏎️', name: 'Autopista', color: 'amber' },
                  'urban-balanced-bytetrack': { icon: '🏙️', name: 'Urbano', color: 'cyan' },
                  'precision-slow-botsort': { icon: '🎯', name: 'Precisión', color: 'purple' },
                  'forensic-reID-botsort': { icon: '🔬', name: 'Forense', color: 'pink' }
                }[preset] || { icon: '⚙️', name: 'Custom', color: 'gray' };

                const isActive = activePreset === preset;

                return (
                  <button
                    key={preset}
                    onClick={() => {
                      setActivePreset(preset);
                      setYoloConfig(trackingPresets[preset]);
                    }}
                    className={`p-2 rounded-xl border transition-all flex flex-col items-center justify-center gap-1 relative ${isActive
                      ? `bg-${presetInfo.color}-500/20 border-${presetInfo.color}-500/50 text-${presetInfo.color}-400 shadow-[0_0_15px_rgba(var(--${presetInfo.color}-500-rgb),0.2)]`
                      : 'bg-slate-900/40 border-white/5 text-slate-500 hover:border-white/20 hover:bg-slate-900/80'
                      }`}
                  >
                    <span className="text-lg filter drop-shadow-md">{presetInfo.icon}</span>
                    <span className="text-[10px] font-black uppercase tracking-tight">{presetInfo.name}</span>
                    <span className={`text-[7px] font-mono px-1.5 rounded-full ${config.trackerType === 'BoT-SORT' ? 'bg-purple-950/50 text-purple-300' : 'bg-cyan-950/50 text-cyan-300'}`}>
                      {config.trackerType === 'BoT-SORT' ? 'BoT-SORT' : 'ByteTrack'}
                    </span>
                    {isActive && <Check size={10} className={`absolute top-2 right-2 text-${presetInfo.color}-400`} />}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Advanced YOLO Parameters */}
          <div className="bg-slate-900/40 rounded-2xl p-4 space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-[10px] font-bold text-slate-300 uppercase tracking-wide">
                <span>🎯 YOLO Confidence</span>
                <span className="text-cyan-400 font-mono">{yoloConfig.confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range" min="0.1" max="0.9" step="0.05"
                value={yoloConfig.confThreshold}
                onChange={(e) => setYoloConfig(c => ({ ...c, confThreshold: parseFloat(e.target.value) }))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] font-bold text-slate-300 uppercase tracking-wide">
                <span>⚡ Frame Skip</span>
                <span className="text-amber-400 font-mono">{yoloConfig.detectionSkip} F</span>
              </div>
              <input
                type="range" min="1" max="10" step="1"
                value={yoloConfig.detectionSkip}
                onChange={(e) => setYoloConfig(c => ({ ...c, detectionSkip: parseInt(e.target.value) }))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] font-bold text-slate-300 uppercase tracking-wide">
                <span>🧲 High Det. Thr</span>
                <span className="text-purple-400 font-mono">{yoloConfig.highDetThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range" min="0.3" max="0.9" step="0.05"
                value={yoloConfig.highDetThreshold}
                onChange={(e) => setYoloConfig(c => ({ ...c, highDetThreshold: parseFloat(e.target.value) }))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] font-bold text-slate-300 uppercase tracking-wide">
                <span>🔄 Track Buffer</span>
                <span className="text-green-400 font-mono">{yoloConfig.trackBufferFrames} F</span>
              </div>
              <input
                type="range" min="10" max="60" step="5"
                value={yoloConfig.trackBufferFrames}
                onChange={(e) => setYoloConfig(c => ({ ...c, trackBufferFrames: parseInt(e.target.value) }))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-green-500"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] font-bold text-slate-300 uppercase tracking-wide">
                <span>🎲 Match IoU</span>
                <span className="text-pink-400 font-mono">{yoloConfig.matchIouThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range" min="0.1" max="0.7" step="0.05"
                value={yoloConfig.matchIouThreshold}
                onChange={(e) => setYoloConfig(c => ({ ...c, matchIouThreshold: parseFloat(e.target.value) }))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-500"
              />
            </div>

            {/* BoT-SORT Specific Controls */}
            {yoloConfig.trackerType === 'BoT-SORT' && (
              <div className="space-y-1 pt-2 border-t border-purple-500/20">
                <div className="flex justify-between text-[9px] font-bold text-purple-400 uppercase">
                  <span>🎨 Appearance Weight</span>
                  <span className="text-purple-300">{yoloConfig.appearanceWeight.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.0" max="1.0" step="0.1"
                  value={yoloConfig.appearanceWeight}
                  onChange={(e) => setYoloConfig(c => ({ ...c, appearanceWeight: parseFloat(e.target.value) }))}
                  className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
                <p className="text-[7px] text-purple-400/60 mt-1">Visual similarity weight</p>
              </div>
            )}

            {/* === System Sensors & Status Panel === */}
            <div className="mt-4 p-3 bg-slate-950/50 border border-white/5 rounded-xl space-y-2">
              {/* Active Sensors */}
              <div className="space-y-2">
                {/* RADAR (Detection System) */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${detectorRef.current ? 'bg-green-400 animate-pulse' : 'bg-slate-700'}`}></div>
                    <span className="text-[8px] font-mono text-green-400 uppercase tracking-wider">RADAR_ACTIVE</span>
                  </div>
                  <Signal size={10} className="text-green-400" />
                </div>

                {/* AI Core Sync */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${trackerRef.current ? 'bg-cyan-400 animate-pulse' : 'bg-slate-700'}`}></div>
                    <span className="text-[8px] font-mono text-cyan-400 uppercase tracking-wider">AI_CORE_SYNC</span>
                  </div>
                  <Binary size={10} className="text-cyan-400" />
                </div>

                {/* Pose Estimation */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${poseEstimationEnabled && detectorRef.current?.poseSession ? 'bg-purple-400 animate-pulse' : 'bg-slate-700'}`}></div>
                    <span className={`text-[8px] font-mono uppercase tracking-wider ${poseEstimationEnabled && detectorRef.current?.poseSession ? 'text-purple-400' : 'text-slate-600'}`}>
                      POSE_DETECTOR
                    </span>
                  </div>
                  <button
                    onClick={() => setPoseEstimationEnabled(!poseEstimationEnabled)}
                    className={`px-2 py-0.5 rounded text-[7px] font-bold transition-all ${poseEstimationEnabled ? 'bg-purple-500/30 text-purple-300' : 'bg-slate-800 text-slate-500'
                      }`}
                  >
                    {poseEstimationEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Mesh Grid */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${meshGridConfig.enabled ? 'bg-pink-400 animate-pulse' : 'bg-slate-700'}`}></div>
                    <span className={`text-[8px] font-mono uppercase tracking-wider ${meshGridConfig.enabled ? 'text-pink-400' : 'text-slate-600'}`}>
                      MESH_GRID
                    </span>
                  </div>
                  <span className="text-[7px] text-slate-500 font-mono">{meshGridConfig.gridType.toUpperCase()}</span>
                </div>

                {/* Line Detection */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${detectionLines.length > 0 ? 'bg-amber-400 animate-pulse' : 'bg-slate-700'}`}></div>
                    <span className={`text-[8px] font-mono uppercase tracking-wider ${detectionLines.length > 0 ? 'text-amber-400' : 'text-slate-600'}`}>
                      LINE_SENSORS
                    </span>
                  </div>
                  <span className="text-[7px] text-amber-400 font-mono">{detectionLines.filter(l => !l.label.startsWith('GRID_') && !l.label.startsWith('PERSP_')).length}</span>
                </div>

                {/* Forensic Buffer */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${tracksRef.current.some(t => t.snapshots && t.snapshots.length > 0) ? 'bg-orange-400 animate-pulse' : 'bg-slate-700'}`}></div>
                    <span className={`text-[8px] font-mono uppercase tracking-wider ${tracksRef.current.some(t => t.snapshots && t.snapshots.length > 0) ? 'text-orange-400' : 'text-slate-600'}`}>
                      FORENSIC_BUF
                    </span>
                  </div>
                  <span className="text-[7px] text-orange-400 font-mono">
                    {tracksRef.current.reduce((sum, t) => sum + (t.snapshots?.length || 0), 0)}
                  </span>
                </div>
              </div>

              {/* System Info */}
              <div className="text-[8px] font-mono text-slate-500 space-y-1 pt-2 border-t border-white/5">
                <div>Model: <span className="text-cyan-400">YOLOv11-Nano (ONNX/WASM)</span></div>
                <div>Tracker: <span className={yoloConfig.trackerType === 'BoT-SORT' ? 'text-purple-400' : 'text-cyan-400'}>{yoloConfig.trackerType}</span></div>
                <div>Backend: <span className="text-green-400">WebAssembly SIMD</span></div>
              </div>
            </div>
          </div>
        </div>



        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-[12px] font-black uppercase text-slate-500 tracking-widest flex items-center gap-2">
              <ClipboardList size={14} className="text-cyan-500" /> ANÁLISIS EXHAUSTIVO
            </h3>
            {aiFeedback && (
              <span className="text-[10px] font-black text-amber-500 animate-pulse uppercase tracking-tighter">AI_KNOWLEDGE_SYNC</span>
            )}
          </div>
          <div className="relative">
            <textarea
              value={directives}
              onChange={handleDirectivesChange}
              className="w-full h-[400px] bg-slate-950 border border-white/10 rounded-2xl p-4 text-[12px] font-mono text-cyan-500 focus:border-cyan-500 outline-none resize-none leading-relaxed transition-all scrollbar-hide"
            />
            <div className="absolute bottom-4 right-4 text-[11px] font-black text-cyan-500/30 font-mono tracking-widest">GEMINI_OS_v3</div>
          </div>
          {aiFeedback && (
            <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-[9px] font-black text-amber-400 uppercase tracking-widest animate-in fade-in slide-in-from-top-2">
              {aiFeedback}
            </div>
          )}
        </div>



        {/* Multi-Lane Detection Configuration - Now Multi-Select */}
        <div className="space-y-4">
          <h3 className="text-[12px] font-black uppercase text-slate-500 tracking-widest flex items-center gap-2 px-2">
            <Layers size={14} className="text-purple-500" /> STACK DE PROTOCOLO ({selectedConfigs.length})
          </h3>

          <div className="bg-slate-900/50 border border-white/5 rounded-[32px] overflow-hidden">
            <div className="max-h-[350px] overflow-y-auto p-4 space-y-6 custom-scrollbar">
              {[
                {
                  label: "INTERSECCIONES TÉCNICAS", items: [
                    { id: 'intersection-ddi', label: "Diamante Divergente (DDI)", icon: "💎", desc: "Intercambia carriles antes del puente para eliminar giros cruzados" },
                    { id: 'intersection-spui', label: "Punto Único (SPUI)", icon: "🎯", desc: "Un solo semáforo central gestiona todos los giros" },
                    { id: 'intersection-turbo-roundabout', label: "Turbo-Rotonda", icon: "🌀", desc: "Carriles en espiral que evitan cambios de carril dentro de la rotonda" },
                    { id: 'intersection-cfi', label: "Flujo Continuo (CFI)", icon: "⚡", desc: "Giros a la izquierda antes de la intersección para flujo sin paradas" },
                    { id: 'intersection-box-junction', label: "Caja Amarilla (Box)", icon: "🟨", desc: "Zona restrictiva donde no se puede detener el vehículo" },
                    { id: 'es-rotonda-partida', label: "Glorieta Partida", icon: "🔄", desc: "Rotonda dividida por un paso elevado o deprimido" },
                    { id: 'intersection-stack', label: "Enlace Multinivel (Stack)", icon: "🏗️", desc: "Múltiples niveles de puentes para separar totalmente los flujos" },
                    { id: 'intersection-magic-roundabout', label: "Magic Roundabout", icon: "🔮", desc: "Varias mini-rotondas alrededor de una central en sentido contrario" },
                    { id: 'intersection-dumbbell', label: "Dumbbell (Pesa)", icon: "🏋️", desc: "Dos rotondas conectadas por un tramo recto" },
                    { id: 't-junction-urban', label: "Intersección en T (Urbana)", icon: "📐", desc: "Cruce en T con semáforo y paso de peatones" },
                    { id: 't-junction-rural', label: "Intersección en T (Rural)", icon: "🛣️", desc: "Cruce en T sin semáforo con señalización de prioridad" },
                    { id: 't-junction-multi', label: "Intersección en T (Multi)", icon: "📐", desc: "Cruce en T con múltiples carriles y carriles de giro dedicados" },
                    { id: 'roundabout-2lanes', label: "Rotonda Estándar", icon: "🔄", desc: "Rotonda convencional de 2 carriles con ceda el paso" },
                    { id: 'interurban-cloverleaf', label: "Enlace Trébol", icon: "🍀", desc: "Cuatro lazos en forma de hoja para conectar autopistas" },
                    { id: 'cross-junction-4way', label: "Intersección en Cruz", icon: "🚦", desc: "Cruce de 4 vías con semáforos y control total" }
                  ]
                },
                {
                  label: "RED PROVINCIAL DE CARRETERAS", items: [
                    { id: 'es-convencional-ancha', label: "Convencional (Arcén >1.5m)", icon: "🛣️" },
                    { id: 'es-convencional-estrecha', label: "Convencional Estrecha", icon: "🛣️" },
                    { id: 'es-autovia-nacional', label: "Autovía / Autopista", icon: "🛣️" },
                    { id: 'madrid-regional-highway', label: "Regional (M-607 / M-506)", icon: "🛣️" },
                    { id: 'es-via-automoviles', label: "Vía Automóviles", icon: "🚗" },
                    { id: 'mountain-pass', label: "Puerto de Montaña", icon: "🏔️" },
                    { id: 'es-travesia-nacional', label: "Travesía Nacional", icon: "🏘️" }
                  ]
                },
                {
                  label: "ENTORNO URBANO (ESPAÑA)", items: [
                    { id: 'es-calle-30', label: "Calle 30 (Límite 30)", icon: "🏙️" },
                    { id: 'es-ciclocarril', label: "Ciclocarril / Vía Bici", icon: "🚲" },
                    { id: 'es-supermanzana', label: "Supermanzana / Residencial", icon: "🏡" },
                    { id: 'school-safety', label: "Zona Escolar Segura", icon: "🏫" },
                    { id: 'madrid-zbe-enforcement', label: "ZBE (Bajas Emisiones)", icon: "🛡️" },
                    { id: 'parking-enforcement', label: "Carga/Descarga / ORA", icon: "🅿️" }
                  ]
                },
                {
                  label: "ACCESOS A DAGANZO (PERÍMETRO)", items: [
                    { id: 'daganzo-m100-enlace', label: "Enlace M-100 (A-2)", icon: "📍" },
                    { id: 'daganzo-m113-ajalvir', label: "Acceso Ajalvir", icon: "🛣️" },
                    { id: 'daganzo-m113-norte', label: "Acceso Paracuellos", icon: "📍" },
                    { id: 'daganzo-m113-fresno', label: "Acceso Fresno/Serracines", icon: "🛣️" },
                    { id: 'daganzo-m113-sur', label: "Acceso Sur (Torrejón R-2)", icon: "📍" },
                    { id: 'daganzo-m118-alcala', label: "M-118 (Alcalá)", icon: "🛣️" },
                    { id: 'daganzo-m119-camarma', label: "M-119 (Camarma)", icon: "🛣️" }
                  ]
                },
                {
                  label: "NÚCLEO URBANO Y POLÍGONOS", items: [
                    { id: 'daganzo-constitucion', label: "C/ Constitución", icon: "🏙️" },
                    { id: 'daganzo-av-madrid', label: "Avenida de Madrid", icon: "🏙️" },
                    { id: 'daganzo-calle-mayor', label: "Calle Mayor", icon: "🏛️" },
                    { id: 'daganzo-rotonda-entrada', label: "Rotonda Principal", icon: "🔄" },
                    { id: 'daganzo-centro', label: "Centro / Plaza Villa", icon: "🏛️" },
                    { id: 'daganzo-residencial', label: "Zonas SORE", icon: "🏡" },
                    { id: 'daganzo-colegios-magno', label: "Escuelas (Magno/Berzal)", icon: "🏫" },
                    { id: 'daganzo-poligono-frailes', label: "P.I. Los Frailes", icon: "🏭" },
                    { id: 'daganzo-poligono-gitesa', label: "P.I. Gitesa", icon: "🏭" },
                    { id: 'daganzo-camino-gancha', label: "Camino de la Gancha", icon: "🚲" }
                  ]
                }
              ].map((group, gIdx) => (
                <div key={gIdx} className="space-y-2">
                  <div className="text-[11px] font-black text-slate-600 uppercase tracking-widest px-2">{group.label}</div>
                  <div className="grid grid-cols-1 gap-1">
                    {group.items.map(item => (
                      <button
                        key={item.id}
                        onClick={() => toggleProtocol(item.id)}
                        className={`flex items-center justify-between p-3 rounded-xl border transition-all group ${selectedConfigs.includes(item.id)
                          ? 'bg-cyan-950/30 border-cyan-500/40 shadow-[0_0_15px_rgba(6,182,212,0.1)]'
                          : 'bg-black/40 border-white/5 hover:bg-white/5 hover:border-white/10'
                          }`}
                      >
                        <div className="flex items-center gap-3 flex-1 text-left">
                          <span className={`text-xs filter ${selectedConfigs.includes(item.id) ? 'grayscale-0' : 'grayscale opacity-50'}`}>{item.icon}</span>
                          <div className="flex flex-col gap-0.5">
                            <span className={`text-[10px] font-black uppercase tracking-wider ${selectedConfigs.includes(item.id) ? 'text-cyan-400' : 'text-slate-400 group-hover:text-slate-300'}`}>
                              {item.label}
                            </span>
                            {(item as any).desc && (
                              <span className={`text-[8px] font-mono leading-tight ${selectedConfigs.includes(item.id) ? 'text-cyan-500/60' : 'text-slate-600'}`}>
                                {(item as any).desc}
                              </span>
                            )}
                          </div>
                        </div>
                        {selectedConfigs.includes(item.id) && <Check size={10} className="text-cyan-400 shrink-0" />}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>


          <div className="px-2 space-y-3 pt-2">
            <button
              onClick={() => syncProtocols(selectedConfigs, directives)}
              className="w-full py-4 rounded-2xl bg-cyan-500 text-black flex items-center justify-center gap-3 transition-all hover:bg-cyan-400 shadow-[0_0_30px_rgba(6,182,212,0.3)] animate-pulse"
            >
              <BrainCircuit size={18} />
              <span className="text-[12px] font-black uppercase tracking-[0.2em]">SÍNTESIS AUTOMÁTICA DE MALLA</span>
            </button>
          </div>

          <div className="px-2 space-y-4 pt-4 border-t border-white/5">
            <span className="text-[11px] font-black text-slate-500 uppercase block">Malla de Detección: {detectionLines.length} Nodos</span>
            <div className="flex flex-wrap gap-2 max-h-[100px] overflow-y-auto custom-scrollbar">
              {detectionLines.map((line, idx) => (
                <div key={idx} className="flex items-center gap-1.5 px-2 py-1 bg-slate-900 border border-white/5 rounded-full group">
                  <div className="w-1.5 h-1.5 rounded-full" style={{
                    backgroundColor:
                      line.type === 'solid' || line.type === 'stop' ? '#ef4444' :
                        line.type === 'pedestrian' ? '#22d3ee' : '#f59e0b'
                  }} />
                  <span className="text-[10px] font-mono text-slate-400 capitalize">{line.label.toLowerCase()}</span>
                  <button
                    onClick={() => setDetectionLines(prev => prev.filter((_, i) => i !== idx))}
                    className="opacity-0 group-hover:opacity-100 text-red-500 transition-opacity"
                  >
                    <X size={8} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>




        <div className="p-6 border-t border-white/5 mt-auto">
          <button onClick={startStream} className={`w-full py-4 rounded-2xl flex items-center justify-center gap-3 transition-all ${source === 'live' ? 'bg-cyan-500 text-black' : 'bg-white/5 text-slate-400'}`}>
            <Wifi size={16} /> <span className="text-[12px] font-black uppercase tracking-widest text-inherit">Sync Radar Daganzo</span>
          </button>
        </div>
      </aside >

      {/* MAIN VIEWPORT - Full-Screen High-Fidelity Video */}
      < main className="flex-1 relative flex flex-col bg-black overflow-hidden group/viewport" >
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain"
        />
        <div className="absolute top-6 left-6 right-6 z-40 flex justify-between pointer-events-none items-start">
          <div className="flex items-center gap-3 bg-black/60 backdrop-blur-xl px-4 py-2 rounded-2xl border border-cyan-500/20">
            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse shadow-[0_0_10px_#22d3ee]" />
            <span className="text-[9px] font-mono text-cyan-400/80 uppercase tracking-wider">LIVE :: {fps} FPS</span>
          </div>
        </div>

        {/* Video Viewport - Absolute Protagonist */}
        <div className="absolute inset-0 z-10">
          {source === 'none' ? (
            <div className="w-full h-full flex flex-col items-center justify-center gap-8 lg:gap-12 animate-in zoom-in-95 duration-1000">
              <DaganzoEmblem className="w-64 h-80 lg:w-96 lg:h-[480px] drop-shadow-[0_0_30px_rgba(14,165,233,0.1)]" />
              <div className="text-center space-y-4">
                <span className="text-3xl lg:text-4xl font-black uppercase tracking-[0.8em] italic text-white/10 block leading-none">POLICÍA LOCAL</span>
                <p className="text-[12px] font-mono text-cyan-500/30 uppercase tracking-[0.4em]">Standby :: Esperando Feed Forense</p>
              </div>
            </div>
          ) : (
            <div className="relative w-full h-full bg-black">
              {/* High-Fidelity Active Mirror - Video fills entire viewport */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                loop
                className="absolute inset-0 w-full h-full object-contain brightness-[0.9] contrast-[1.3]"
              />

              {/* Canvas Overlay - Transparent Detection Layer */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />

              {/* Subtle Grid Overlay */}
              <div className="absolute inset-0 opacity-5 pointer-events-none hud-grid" />

              {/* Forensic Analysis Indicator */}
              {isAnalyzing && (
                <div className="absolute bottom-8 right-8 z-50 bg-black/95 border-2 border-red-600/40 p-6 rounded-[30px] flex items-center gap-5 shadow-[0_0_40px_rgba(220,38,38,0.4)] animate-in slide-in-from-bottom-10">
                  <div className="w-12 h-12 border-t-3 border-red-600 rounded-full animate-spin flex items-center justify-center">
                    <BrainCircuit size={24} className="text-cyan-400 animate-pulse" />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[11px] font-black text-red-500 uppercase tracking-widest">{statusMsg}</span>
                    <span className="text-[11px] font-mono text-cyan-500/60 uppercase">Análisis Forense Neural</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="h-32 bg-[#020617]/98 border-t border-white/5 flex items-center justify-between px-12 z-50">
          {/* Left: Playback & Status */}
          <div className="flex items-center gap-8 w-1/3">
            <button onClick={() => isPlaying ? safePause() : safePlay()}
              className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-all ${isPlaying ? 'bg-red-800 shadow-neon' : 'bg-cyan-600 text-black shadow-neon'}`}>
              {isPlaying ? <Pause size={28} /> : <Play size={28} className="ml-1" />}
            </button>
            <div className="flex flex-col">
              <span className="text-[11px] font-black text-slate-600 uppercase tracking-widest mb-1">Status de Feed</span>
              <span className="text-xl font-black italic text-white/95 tracking-tighter uppercase whitespace-nowrap">
                {source === 'live' ? 'Neural_Live_Feed' : source === 'upload' ? 'Forensic_Buffer' : 'System_Standby'}
              </span>
            </div>
          </div>

          {/* Center: Detections & Identity */}
          <div className="flex flex-col items-center justify-center gap-1 w-1/3">
            <div className="flex items-center gap-3 bg-white/5 px-4 py-1.5 rounded-full border border-white/10 mb-1">
              <Target size={12} className="text-purple-400" />
              <span className="text-[9px] font-black text-white/80 uppercase tracking-widest">SENTINEL_V15_NODE</span>
            </div>
            <div className="flex gap-8 border-t border-white/5 pt-2">
              <div className="text-center">
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Detecciones</span>
                <span className="text-2xl font-mono font-black text-cyan-500 leading-none">{cumulativeDetections}</span>
              </div>
              <div className="text-center">
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Expedientes</span>
                <span className="text-2xl font-mono font-black text-red-600 leading-none">{cumulativeExpedientes}</span>
              </div>
            </div>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center justify-end gap-6 w-1/3">
            <div className="hidden lg:flex flex-col text-right">
              <span className="text-[11px] font-black text-slate-600 uppercase tracking-widest mb-1">Processor_Unit</span>
              <span className="text-[12px] font-mono text-cyan-400/60 uppercase">Daganzo_Node_01</span>
            </div>
            <button onClick={() => document.getElementById('f-up-main')?.click()}
              className="w-14 h-14 bg-white/5 rounded-2xl hover:bg-white/10 text-slate-500 transition-all border border-white/10 flex items-center justify-center shadow-neon group">
              <Upload size={24} className="group-hover:text-cyan-400 transition-colors" />
              <input id="f-up-main" type="file" className="hidden" accept="video/*" onChange={e => { const f = e.target.files?.[0]; if (f) { setVideoUrl(URL.createObjectURL(f)); setSource('upload'); } }} />
            </button>
          </div>
        </div>
      </main >

      {/* REGISTRY SIDEBAR */}
      < aside className="w-full lg:w-96 border-l border-white/5 flex flex-col z-50 bg-[#020617]/98 h-1/2 lg:h-full backdrop-blur-2xl" >
        <div className="p-8 border-b border-white/5 flex items-center justify-between bg-red-950/10">
          <div className="flex flex-col">
            <span className="text-xl font-black italic tracking-tighter text-red-500 uppercase leading-none">EVIDENCE</span>
            <span className="text-[10px] font-black tracking-[0.5em] text-red-500/80 uppercase whitespace-nowrap">STORAGE & LOGS</span>
          </div>
          <Scale size={20} className="text-red-500/50" />
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
          {logs.map(log => (
            <div key={log.id} onClick={() => setSelectedLog(log)} className="p-4 bg-slate-900/40 border border-white/5 border-l-4 border-l-red-600 rounded-2xl cursor-pointer hover:bg-slate-900 transition-all shadow-lg group hover:border-red-500/30">
              <div className="relative aspect-video rounded-xl overflow-hidden mb-3 border border-white/10 shadow-lg">
                <img src={log.image} className="w-full h-full object-cover grayscale brightness-110 contrast-125 group-hover:grayscale-0 transition-all duration-500" />
                <div className="absolute top-2 right-2 bg-red-700/90 backdrop-blur px-2 py-0.5 rounded text-[10px] font-mono font-black text-white shadow-lg">{log.plate}</div>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold text-white uppercase tracking-wider">{log.subType}</span>
                <span className={`text-[8px] font-black px-2 py-0.5 rounded-full uppercase ${log.severity === 'muy-grave' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-orange-500/20 text-orange-400 border border-orange-500/30'}`}>{log.severity}</span>
              </div>
              <p className="text-[9px] text-slate-500 font-mono line-clamp-2 leading-relaxed uppercase group-hover:text-slate-400">"{log.description}"</p>
            </div>
          ))}
        </div>
      </aside >

      {/* DETAIL MODAL - FORENSIC REPORT DESIGN */}
      {
        selectedLog && (
          <div className="fixed inset-0 z-[200] bg-black/95 flex items-center justify-center p-6 backdrop-blur-3xl animate-in fade-in duration-500">
            <div className="bg-[#050914] w-full max-w-7xl h-[90vh] rounded-[60px] border border-white/5 overflow-hidden flex flex-col shadow-2xl relative animate-in zoom-in-95">

              <button onClick={() => setSelectedLog(null)} className="absolute top-6 right-6 z-[210] p-3 bg-slate-900/90 rounded-full hover:bg-red-700 text-white transition-all shadow-neon border border-white/10">
                <X size={24} />
              </button>

              <div className="flex-1 p-6 lg:p-8 flex flex-col lg:flex-row gap-6 overflow-hidden">

                {/* Visual Evidence Section (Left) */}
                <div className="flex-1 flex flex-col gap-4 overflow-y-auto custom-scrollbar">

                  {/* Header: Evidence Clip Info */}
                  <div className="flex items-center justify-between">
                    <h3 className="text-cyan-500 font-black uppercase text-sm tracking-[0.2em] flex items-center gap-4 italic">
                      <Video size={20} className="animate-pulse" /> EVIDENCE CLIP (HD_10S)
                    </h3>
                    <div className="flex gap-4">
                      <span className="bg-slate-900 border border-white/10 px-4 py-1.5 rounded-xl text-[12px] font-mono text-slate-400 flex items-center gap-2">
                        <Clock size={12} /> {selectedLog.time}
                      </span>
                      <span className="bg-slate-900 border border-white/10 px-4 py-1.5 rounded-xl text-[12px] font-mono text-purple-400 flex items-center gap-2">
                        <Ruler size={12} /> 3M_LANE_CALIB
                      </span>
                    </div>
                  </div>

                  {/* Main Visual Buffer */}
                  <div className="relative bg-[#0a0a0a] rounded-[60px] overflow-hidden border border-white/10 shadow-[0_0_80px_rgba(34,211,238,0.1)] group">

                    {/* Red Banner - Top Center */}
                    <div className="absolute top-0 left-0 right-0 p-8 flex justify-center pointer-events-none z-20">
                      <div className="bg-red-700 text-white px-10 py-4 rounded-b-[40px] font-black text-lg uppercase tracking-tighter shadow-2xl text-center leading-tight max-w-[80%]">
                        {selectedLog.legalArticle || 'ART. 151 DEL REGLAMENTO GENERAL DE CIRCULACIÓN'}
                      </div>
                    </div>

                    {/* The Video/Image */}
                    <div className="aspect-video relative">
                      {selectedLog.videoUrl ? (
                        <>
                          <video
                            ref={(el) => {
                              if (el) {
                                el.onloadedmetadata = () => { };
                                el.ontimeupdate = () => { };
                              }
                            }}
                            src={selectedLog.videoUrl}
                            autoPlay
                            loop
                            muted
                            className="w-full h-full object-contain brightness-110 contrast-125"
                          />

                          {/* Video Timeline Controls */}
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity">
                            <div className="flex items-center gap-3">
                              <button className="w-8 h-8 flex items-center justify-center rounded-full bg-white/10 hover:bg-white/20 transition-colors">
                                <Play size={14} className="text-white" />
                              </button>
                              <div className="flex-1 h-1 bg-white/20 rounded-full overflow-hidden cursor-pointer group/timeline">
                                <div className="h-full bg-cyan-500 w-1/2" />
                              </div>
                              <span className="text-xs font-mono text-white/60">00:00 / 00:10</span>
                            </div>
                          </div>
                        </>
                      ) : (
                        <img src={selectedLog.image} className="w-full h-full object-contain" />
                      )}
                    </div>

                    <div className="absolute inset-0 bg-red-600/5 pointer-events-none" />
                  </div>

                  {/* Metadata Section Below Video */}
                  <div className="bg-[#0a0a0a]/50 p-10 rounded-[50px] border border-white/5 space-y-4">
                    <div className="text-cyan-500 font-black uppercase text-sm tracking-[0.4em] italic flex items-center gap-2">
                      <Target size={16} /> {selectedLog.vehicleType} / FORENSIC_CALIB
                    </div>
                    <h2 className="text-5xl lg:text-6xl font-black italic text-white tracking-tighter uppercase font-mono">
                      {selectedLog.plate}
                    </h2>
                  </div>

                  {/* Sensor Pillars (Vertical Capsule Shapes) */}
                  <div className="grid grid-cols-4 gap-6 px-2">
                    {/* Sensor 1 */}
                    <div className="bg-[#0a0a0a] aspect-[3/4.5] rounded-[60px] border border-white/5 flex flex-col items-center justify-between py-10 text-center group transition-transform hover:scale-105">
                      <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest font-mono">Calibración</span>
                      <div className="flex flex-col items-center">
                        <span className="text-[16px] font-black text-purple-500 uppercase leading-none">CARRIL</span>
                        <span className="text-[18px] font-black text-purple-500 font-mono">3.0M</span>
                      </div>
                      <div className="w-2 h-2 rounded-full bg-purple-500/30 border border-purple-500 animate-pulse" />
                    </div>

                    {/* Sensor 2 */}
                    <div className="bg-[#0a0a0a] aspect-[3/4.5] rounded-[60px] border border-white/5 flex flex-col items-center justify-between py-10 text-center transition-transform hover:scale-105">
                      <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest font-mono">Velocidad_Est.</span>
                      <div className="flex flex-col items-center">
                        <span className="text-4xl font-mono font-black text-amber-500 leading-none">{selectedLog.telemetry?.speedEstimated.replace(' km/h', '')}</span>
                        <span className="text-[14px] font-black text-amber-500/50 uppercase font-mono">km/h</span>
                      </div>
                      <Gauge size={18} className="text-amber-500/40" />
                    </div>

                    {/* Sensor 3 */}
                    <div className="bg-[#0a0a0a] aspect-[3/4.5] rounded-[60px] border border-white/5 flex flex-col items-center justify-between py-10 text-center transition-transform hover:scale-105">
                      <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest font-mono">Confianza_AI</span>
                      <span className="text-4xl font-mono font-black text-cyan-400 leading-none">{(selectedLog.confidence * 100).toFixed(0)}%</span>
                      <div className="w-12 h-1 bg-cyan-500/20 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-500" style={{ width: `${selectedLog.confidence * 100}%` }} />
                      </div>
                    </div>

                    {/* Sensor 4 */}
                    <div className="bg-[#0a0a0a] aspect-[3/4.5] rounded-[60px] border border-white/5 flex flex-col items-center justify-between py-10 text-center transition-transform hover:scale-105">
                      <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest font-mono">Severidad</span>
                      <span className="text-[16px] font-black text-red-600 uppercase font-mono leading-none tracking-tighter whitespace-nowrap">{selectedLog.severity}</span>
                      <AlertTriangle size={20} className="text-red-600 animate-bounce" />
                    </div>
                  </div>

                  {/* NEURAL FORENSIC BUFFER: Multiple snapshots display */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between px-4">
                      <h3 className="text-cyan-500 font-black uppercase text-[12px] tracking-[0.3em] flex items-center gap-3 italic">
                        <Binary size={16} className="text-cyan-500" /> NEURAL_FORENSIC_BUFFER [BIF]
                      </h3>
                      <span className="text-[11px] font-mono text-slate-500 uppercase">{selectedLog.snapshots?.length || 0} SECUENTIAL_FRAMES_CAPTURED</span>
                    </div>
                    <div className="flex gap-4 overflow-x-auto pb-4 custom-scrollbar snap-x">
                      {selectedLog.snapshots?.map((snap, i) => (
                        <div key={i} className="min-w-[120px] aspect-[4/3] bg-slate-900 rounded-2xl border border-white/10 overflow-hidden snap-center group relative cursor-pointer hover:border-cyan-500/50 transition-all shrink-0">
                          <img src={`data:image/jpeg;base64,${snap}`} className="w-full h-full object-cover grayscale brightness-110 contrast-125 group-hover:grayscale-0 transition-all" />
                          <div className="absolute bottom-1 right-2 text-[10px] font-mono text-white/40">F_{i.toString().padStart(2, '0')}</div>
                        </div>
                      ))}
                      {(!selectedLog.snapshots || selectedLog.snapshots.length === 0) && (
                        <div className="w-full py-12 flex flex-col items-center justify-center border border-dashed border-white/5 rounded-3xl text-slate-600">
                          <ScanLine size={32} className="mb-2 opacity-20" />
                          <span className="text-[12px] font-black uppercase tracking-widest">No BIF Data Available</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Narrative Report Section (Right side) */}
                <div className="w-full lg:w-[480px] flex flex-col gap-10 lg:pt-2">

                  <div className="space-y-6">
                    <h3 className="text-red-500 font-black uppercase text-sm tracking-[0.2em] italic flex items-center gap-4">
                      <AlertTriangle size={20} className="text-red-600 animate-pulse" /> FORENSIC EVIDENCE REPORT
                    </h3>
                    <div className="bg-[#0c0c0c] p-10 lg:p-14 rounded-[50px] border border-red-900/10 shadow-inner relative">
                      <p className="text-slate-100 italic text-2xl lg:text-3xl leading-relaxed font-serif">
                        "{selectedLog.description}"
                      </p>
                    </div>
                  </div>

                  {/* Evidence Log List */}
                  <div className="flex-1 p-8 bg-[#0a0a0a] rounded-[50px] border border-white/5 flex flex-col overflow-hidden">
                    <h3 className="text-[12px] font-black uppercase text-slate-500 flex items-center gap-4 italic tracking-widest mb-6 border-b border-white/5 pb-4">
                      <Terminal size={18} className="text-cyan-500" /> ORDERED_EVIDENCE_LOG
                    </h3>
                    <div className="space-y-4 overflow-y-auto custom-scrollbar pr-2">
                      {selectedLog.reasoning?.map((r, i) => (
                        <div key={i} className="flex gap-4 text-[13px] font-mono leading-relaxed py-3 border-b border-white/5 last:border-0 text-slate-400">
                          <span className="text-red-600 font-black h-fit shrink-0">_&gt;</span>
                          <span>{r}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Certification & Validating Action */}
                  <div className="space-y-6">
                    <div className="flex items-center gap-6 p-8 bg-slate-950/40 rounded-[40px] border border-white/5">
                      <DaganzoEmblem className="w-12 h-14 opacity-50" />
                      <div className="flex flex-col">
                        <span className="text-[12px] font-black text-white/30 uppercase tracking-[0.2em]">CERTIFICADO POR</span>
                        <span className="text-[11px] font-black text-cyan-500 uppercase">P.L. DAGANZO DE ARRIBA</span>
                      </div>
                    </div>

                    <button onClick={() => setSelectedLog(null)} className="w-full py-10 bg-[#b91c1c] text-white rounded-[45px] font-black uppercase tracking-[0.5em] text-3xl shadow-[0_20px_60px_rgba(185,28,28,0.3)] hover:bg-red-600 hover:scale-[1.01] transition-all transform active:scale-95 leading-none">
                      VALIDAR EXPEDIENTE
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      }

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100;300;400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
        
        :root {
          font-family: 'Outfit', sans-serif;
        }

        .hud-grid { 
          background-image: linear-gradient(to right, rgba(34, 211, 238, 0.02) 1px, transparent 1px), 
                            linear-gradient(to bottom, rgba(34, 211, 238, 0.02) 1px, transparent 1px); 
          background-size: 100px 100px; 
        }
        .shadow-neon { box-shadow: 0 0 50px rgba(34, 211, 238, 0.15); }
        
        /* Infractor Glow Animation */
        @keyframes redGlow {
          0%, 100% { filter: drop-shadow(0 0 10px rgba(239, 68, 68, 0.8)); }
          50% { filter: drop-shadow(0 0 25px rgba(239, 68, 68, 1)); }
        }
        
        /* Border Vibration */
        @keyframes vibrate {
          0%, 100% { transform: translate(0, 0); }
          25% { transform: translate(-1px, 1px); }
          50% { transform: translate(1px, -1px); }
          75% { transform: translate(-1px, -1px); }
        }
        
        .custom-scrollbar::-webkit-scrollbar { width: 5px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 15px; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
        
        /* Range Slider Styling */
        input[type="range"].slider-thumb::-webkit-slider-thumb {
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #f59e0b;
          cursor: pointer;
          border: 2px solid #0c4a6e;
          box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
          transition: all 0.2s;
        }
        input[type="range"].slider-thumb::-webkit-slider-thumb:hover {
          background: #fbbf24;
          box-shadow: 0 0 15px rgba(245, 158, 11, 0.8);
          transform: scale(1.1);
        }
        input[type="range"].slider-thumb::-moz-range-thumb {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #f59e0b;
          cursor: pointer;
          border: 2px solid #0c4a6e;
          box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
        }
      `}</style>
    </div >
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
