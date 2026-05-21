/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Manrope', 'system-ui', 'sans-serif'],
      },
      colors: {
        ink: '#082032',
        aqua: '#0a9396',
        teal: '#005f73',
      },
      boxShadow: {
        panel: '0 10px 30px rgba(8, 32, 50, 0.08)',
      },
    },
  },
  plugins: [],
}
