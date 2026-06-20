using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using KomatsuIntel.Frontend;
using KomatsuIntel.Frontend.Services;
using MudBlazor.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.Services.AddScoped(sp => new HttpClient
{
    BaseAddress = new Uri(builder.Configuration["BackendBaseUrl"] ?? "http://localhost:8000/")
});

builder.Services.AddMudServices();
builder.Services.AddScoped<ApiClient>();

await builder.Build().RunAsync();
