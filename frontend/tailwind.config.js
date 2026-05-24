/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ocean: {
          50:  '#edf8ff',
          100: '#d6efff',
          200: '#a8dcff',
          300: '#64c2ff',
          400: '#1aa3ff',
          500: '#0082e6',
          600: '#0062c4',
          700: '#004d9e',
          800: '#003d82',
          900: '#00306b',
          950: '#001a45',
        },
        brutum: {
          black:  '#0a0e1a',
          dark:   '#0f1629',
          mid:    '#1a2444',
          border: '#2a3a6e',
          accent: '#00d4ff',
          yellow: '#ffe600',
          coral:  '#ff4d6d',
          green:  '#00ff9f',
        }
      },
      fontFamily: {
        display: ['"Space Mono"', 'monospace'],
        body:    ['"IBM Plex Mono"', 'monospace'],
        sans:    ['"IBM Plex Sans"', 'sans-serif'],
      },
      boxShadow: {
        brutal:     '4px 4px 0px #00d4ff',
        'brutal-lg':'6px 6px 0px #00d4ff',
        'brutal-xl':'8px 8px 0px #00d4ff',
        'brutal-y': '4px 4px 0px #ffe600',
        'brutal-r': '4px 4px 0px #ff4d6d',
        'brutal-g': '4px 4px 0px #00ff9f',
        inner:      'inset 2px 2px 0px rgba(0,212,255,0.2)',
      },
      borderWidth: { 3: '3px' },
      animation: {
        'pulse-slow':    'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'slide-in':      'slideIn 0.3s ease-out',
        'fade-up':       'fadeUp 0.4s ease-out',
        'blink':         'blink 1s step-end infinite',
        'scan':          'scan 2s linear infinite',
      },
      keyframes: {
        slideIn:  { from:{ transform:'translateX(-100%)', opacity:'0' }, to:{ transform:'translateX(0)', opacity:'1' } },
        fadeUp:   { from:{ transform:'translateY(20px)', opacity:'0' }, to:{ transform:'translateY(0)', opacity:'1' } },
        blink:    { '0%,100%':{ opacity:'1' }, '50%':{ opacity:'0' } },
        scan:     { from:{ backgroundPosition:'0 0' }, to:{ backgroundPosition:'0 100%' } },
      }
    },
  },
  plugins: [],
}
