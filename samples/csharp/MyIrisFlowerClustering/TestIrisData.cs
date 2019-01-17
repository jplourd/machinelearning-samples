namespace IrisFlowerClustering
{
    // <SnippetStatic>
    static class TestIrisData
    // </SnippetStatic>
    {
        // Single instance of data to test with


        // <SnippetTestData>
        internal static readonly IrisData Setosa = new IrisData
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };
        // </SnippetTestData>
    }
}
