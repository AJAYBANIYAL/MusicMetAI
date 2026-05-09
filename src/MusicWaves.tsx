/**
 * Bright music-style layered waves — gradients only, no Swiss grid graphics.
 */

type SvgProps = { className?: string };

/** Decorative wave stack for hero / side panels */
export function MusicWaveStack({ className }: SvgProps) {
  return (
    <svg
      viewBox="0 0 420 460"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <linearGradient id="mwG1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="35%" stopColor="#c084fc" />
          <stop offset="70%" stopColor="#f472b6" />
          <stop offset="100%" stopColor="#fb923c" />
        </linearGradient>
        <linearGradient id="mwG2" x1="100%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="45%" stopColor="#a78bfa" />
          <stop offset="100%" stopColor="#fbbf24" />
        </linearGradient>
        <linearGradient id="mwG3" x1="50%" y1="0%" x2="50%" y2="100%">
          <stop offset="0%" stopColor="#4ade80" />
          <stop offset="50%" stopColor="#34d399" />
          <stop offset="100%" stopColor="#2dd4bf" />
        </linearGradient>
        <linearGradient id="mwG4" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#6366f1" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#ec4899" stopOpacity="0.85" />
        </linearGradient>
      </defs>

      <path
        d="M-20 120 C80 80 140 200 260 165 S420 280 460 310 L460 460 L-20 460 Z"
        fill="url(#mwG1)"
        opacity={0.85}
      />
      <path
        d="M-20 200 C120 260 220 140 340 205 S460 380 470 460 L470 470 L-20 470 Z"
        fill="url(#mwG2)"
        opacity={0.75}
      />
      <path
        d="M-20 300 C90 340 160 260 260 295 S390 245 470 295 L470 460 L-20 460 Z"
        fill="url(#mwG3)"
        opacity={0.55}
      />
      {/* Flowing outline waves */}
      <path
        d="M0 180 Q105 220 210 178 T420 165"
        stroke="url(#mwG4)"
        strokeWidth="8"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M15 238 Q135 278 248 226 T418 258"
        stroke="#fef08a"
        strokeOpacity={0.9}
        strokeWidth="6"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M28 296 Q148 342 278 294 T418 332"
        stroke="#38bdf8"
        strokeOpacity={0.85}
        strokeWidth="5"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M40 360 Q172 392 294 362 T418 394"
        stroke="#f97316"
        strokeOpacity={0.8}
        strokeWidth="4"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

/** Slim colorful wave strip for section breaks */
export function MusicWaveRibbon({ className }: SvgProps) {
  return (
    <svg
      viewBox="0 0 1200 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="rbGrad" x1="0%" y1="50%" x2="100%" y2="50%">
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="25%" stopColor="#e879f9" />
          <stop offset="50%" stopColor="#fb7185" />
          <stop offset="75%" stopColor="#fcd34d" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <path
        d="M0 32 Q150 12 300 28 T600 26 T900 34 T1200 22 V48 H0 Z"
        fill="url(#rbGrad)"
        opacity={0.92}
      />
      <path
        d="M0 40 Q200 24 400 36 T800 30 T1200 38"
        stroke="#fff"
        strokeOpacity={0.35}
        strokeWidth="2"
        fill="none"
      />
    </svg>
  );
}
