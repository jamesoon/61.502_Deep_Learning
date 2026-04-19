const VERSION = import.meta.env.VITE_APP_VERSION || "1.0.0";

export default function Footer() {
  return (
    <footer className="border-t border-gray-200 bg-white py-4 px-6">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-400">
        <span>© 2026 SUTD MSTR-DAIE Deep Learning Project. All rights reserved.</span>
        <div className="flex items-center gap-4">
          <span>Version {VERSION}</span>
          <span>By James Oon | Josiah Lau | Nguyen Tung</span>
        </div>
      </div>
    </footer>
  );
}
